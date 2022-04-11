# This is a rewrite and extension of the repo "simple_blackbox_attack" (https://github.com/cg563/simple-blackbox-attack)
#   which implements the SimBA method in the paper "Simple Black-box Adversarial Attacks"
#   (https://arxiv.org/abs/1905.07121).
# This expands its functionality to generalise the data and models used, and, crucially, allows for experimentation with
#   arbitrary basis vectors.

import configargparse
import math
from pathlib import Path

from scipy.fftpack import idct

import torch

import src.mufasa_utils as muffy
from src.str_parsing import clean_model_name


# It's assumed here that for the pixel basis (trivial identity transformation), the input and attack bases are always of
#   the same dimension (and the trivial identity between the basis vectors holds).
def pixel_basis_meta_fun(input_size):
    def pixel_basis_fun(ind):
        image_dim = torch.tensor(input_size).prod()
        basis_vec = torch.zeros(image_dim)
        basis_vec[ind] = 1
        return basis_vec.view(input_size)
    return pixel_basis_fun


# This is a replication of the low-frequency DCT basis in the original SimBA code. Here, there is an intrinsic
#   dimension of the attack space which will in general be lower than that of the canonical image space. There is a very
#   specific fixed meaning to the indices in this context, defined by the implementation.
def dct_lf_basis_meta_fun(input_size, attack_size):
    def dct_lf_basis_fun(ind):
        attack_dims = torch.tensor(attack_size).prod()
        basis_vec = torch.zeros(attack_dims)
        basis_vec[ind] = 1
        basis_image = basis_vec.view(attack_size)
        padded_basis_image = torch.zeros(input_size)
        padded_basis_image[:, :attack_size[1], :attack_size[2]] = basis_image
        dct_basis_vec = torch.from_numpy(idct(idct(padded_basis_image.numpy(), axis=2, norm='ortho'),
                                              axis=1, norm='ortho'))
        return dct_basis_vec
    return dct_lf_basis_fun


# This represents the option of computing an ordered/scored basis offline, saving it to disc, and reading it here.
# The current usage is on the output of dominant_directions.py, but it can in principle work with any orthonormal
#   basis you want to supply for whatever reason, however you came by it. As I've coded it right now though, it's
#   assuming that your saved file contains two things, the first one being the matrix (the more specific assumption
#   being that load returns (u, s) from the SVD).
# First, I'm implementing a simple version that doesn't worry about how enormous the representation of a basis spanning
#   image space might be, depending on the resolution of image space. It assumes that the column matrix of basis vectors
#   can be read in at once. This may blow up in actual practical applications, and dealing with that is a to-do.
#   However, we will often be dealing with the span of a subspace of image space, i.e. the m-x-n output of a reduced
#   SVD. The actual column dimension of the basis matrix is returned to the client, who can then easily avoid supplying
#   OOB indices, as with the other functions.
def enlo_basis_meta_fun(input_size, basis_path):
    # We're not making use of s in this version, just u. But s can be handy, so it's there.
    svd_stuff = torch.load(basis_path)
    basis_vecs = svd_stuff['u']
    column_dim = basis_vecs.size(1)

    def enlo_basis_fun(ind):
        return basis_vecs[:, ind].view(input_size)
    return enlo_basis_fun, column_dim


# Note that "labels" has a dual identity, in order to allow the same function to be used for targeted and untargeted
#   attacks. If targeted is False, then the labels represent the original (GT) labels, which the attacker must get the
#   network to stop predicting: the predicted labels must not equal the supplied labels. If targeted is True, it's the
#   other way around: the attack is considered successful if and only if the net predicts the supplied label, which will
#   generally be something other than the ground truth (though that is to the client to determine).
# This method assumes that the 'images' tensor is delivered with the gamut expressed in [0,1], and will clamp
#   its own candidate perturbations to this range. Returned attacks assume this convention.
def simba_core(model, normalisation_func, images, labels, attack_basis_generator, attack_basis_inds, targeted, epsilon,
               norm_bound, apply_softmax, subtract_competition, attack_mode, query_saver, batch_size=25,
               return_adv_tensor=False):

    num_images = len(images)
    image_size = images.size()[1:]
    num_batches = math.ceil(num_images / batch_size)
    attack_space_dim = attack_basis_inds.size(1)

    # Because the full stack of adversarial images might be very big, and because it's possible to reconstruct it from
    #   its canonical representation, it is only constructed on request. Otherwise, you get a dummy return.
    if return_adv_tensor:
        adv = torch.Tensor()
    else:
        adv = None
    succs = torch.BoolTensor()
    direction_taken_count = torch.LongTensor()
    query_count = torch.LongTensor()

    # The main return type is a sparse matrix representing the nonzero coefficients of basis vectors used in the
    #   attacks. The "actual" (image-space) attacks can straightforwardly be reconstructed from these by the client
    #   using the basis vector generator function. The client can also request the images directly, but can opt out of
    #   this for space reasons. This representation will in general be far more compact, and allows the client to
    #   moderate memory usage during reconstruction.
    # The sparse matrix construction uses COO format (which should be offered in scipy as well). The Tensors that define
    #   the matrix are thus accumulated accordingly.
    # WARNING: The PyTorch documentation refers to the sparse API as being "in beta", and notes that it "may change in
    #   the near future". This should never distract from the fact that this conceptually can and should be a sparse
    #   matrix.
    canonical_sp_inds, canonical_sp_vals = torch.LongTensor(), torch.Tensor()
    canonical_sp_size = torch.Size((num_images, attack_space_dim))

    for i in range(num_batches):
        print(f"Batch: {i}\n")
        batch_begin_ind, batch_end_ind = (i * batch_size), min((i + 1) * batch_size, num_images)
        image_batch, label_batch = images[batch_begin_ind:batch_end_ind], labels[batch_begin_ind:batch_end_ind]

        basis_sum_images = torch.zeros(image_batch.size())
        if return_adv_tensor:
            adv_pert_images = torch.zeros_like(basis_sum_images)

        # This is a convenience container for doing (L2) rescaling of the existing perturbation and the candidate
        #   direction, to support the L2 ball evaluation.
        batch_direction_taken_count = torch.zeros(len(basis_sum_images)).long()
        batch_query_count = torch.zeros(len(basis_sum_images)).long()

        # These are the COO-format sparse matrix components, for returning the canonical attack representations
        # space-efficiently.
        forward_attack_batch_row_inds, backward_attack_batch_row_inds = torch.LongTensor(), torch.LongTensor()
        forward_attack_batch_col_inds, backward_attack_batch_col_inds = torch.LongTensor(), torch.LongTensor()

        batch_class_preds, batch_label_probs = muffy.get_preds_and_scores(
            model, image_batch, label_batch, normalisation_func, apply_softmax=apply_softmax, targeted_scores=targeted,
            subtract_competition=subtract_competition, batch_size=batch_size
        )

        # Labels mean different things, and we're looking to push probabilities in different directions, depending on
        #   whether the attack is targeted or untargeted.
        # In the targeted case, labels represent targeted classes, and "improvement" in the attack means increasing
        #   those probabilities. An image is on our list of things to do today if its label doesn't equal the target.
        # In the untargeted case, labels represent ground truth, and "improvement" in the attack means decreasing those
        #   probabilities. We're not finished if the predicted label still equals the ground truth.
        # label_comparator takes predicted labels first and supplied labels second, and marks images to be done (True).
        # prob_comparator takes a matrix of stacked probabilities and runs max (targeted) or min (un-) on them to pick
        #   the winner.
        # (Note that you could make this nicer by fixing the sign convention in get_preds_and_scores in mufasa_utils,
        #   which would obviate the below.)
        if targeted:
            label_comparator = torch.ne
            prob_selector = lambda mat: torch.max(mat, 0)
        else:
            label_comparator = torch.eq
            prob_selector = lambda mat: torch.min(mat, 0)

        remaining_image_indices = label_comparator(batch_class_preds, label_batch).nonzero().flatten()

        assert len(remaining_image_indices) == len(label_batch), \
            "You cannot pass any images that are considered to have already been successfully attacked."

        basis_id = 0
        while remaining_image_indices.numel() > 0 and basis_id < attack_space_dim:
            remaining_image_count = len(remaining_image_indices)
            remaining_direction_taken_count = batch_direction_taken_count[remaining_image_indices]
            remaining_query_count = batch_query_count[remaining_image_indices]

            this_basis_vector_stack = torch.zeros([remaining_image_count] + list(image_size))
            # A loop. Not super Pythonic. It is what it is, for now.
            # (Your alternative, or one of them, is to modify all attack basis generator functions to accept vectors
            #   and return stacks.)
            for v in range(remaining_image_count):
                this_basis_vector_stack[v, :, :, :] = attack_basis_generator(
                    attack_basis_inds[remaining_image_indices[v], basis_id]
                )

            current_basis_sum_images = basis_sum_images[remaining_image_indices]
            if return_adv_tensor:
                current_adv_pert_images = adv_pert_images[remaining_image_indices]

            forward_basis_sum_candidates = current_basis_sum_images + this_basis_vector_stack
            backward_basis_sum_candidates = current_basis_sum_images - this_basis_vector_stack

            if attack_mode == 'linf_rescale':
                forward_max_mags = forward_basis_sum_candidates.abs().amax(dim=(1, 2, 3,))
                backward_max_mags = backward_basis_sum_candidates.abs().amax(dim=(1, 2, 3,))
                forward_perts = forward_basis_sum_candidates / forward_max_mags[..., None, None, None] * epsilon
                backward_perts = backward_basis_sum_candidates / backward_max_mags[..., None, None, None] * epsilon
            elif attack_mode == 'linf_signs':
                forward_perts = epsilon * forward_basis_sum_candidates.sign()
                backward_perts = epsilon * backward_basis_sum_candidates.sign()
            elif attack_mode == 'standard_simba':
                # Else, this is the usual SimBA "eps specifies the perturbation size at each iteration" setup, which
                #   is equivalent to eps as the radius of an L-inf ball in the case of the pixel basis (or when the
                #   perturbed image is expressed in terms of whatever basis is being used).
                # *However*, we add a PGD-type option here in which, if a norm_bound (L2) has been specified, then the
                #   total perturbation will be kept within the corresponding ball in the usual manner.
                forward_perts = epsilon * forward_basis_sum_candidates
                backward_perts = epsilon * backward_basis_sum_candidates
                forward_pert_norms = forward_perts.norm(dim=(1, 2, 3,), p=2)
                backward_pert_norms = backward_perts.norm(dim=(1, 2, 3,), p=2)
                forward_perts[forward_pert_norms > norm_bound] /= (
                    forward_pert_norms[forward_pert_norms > norm_bound, None, None, None] / norm_bound)
                backward_perts[backward_pert_norms > norm_bound] /= (
                    backward_pert_norms[backward_pert_norms > norm_bound, None, None, None] / norm_bound)
            else:  # <- invalid attack mode option
                raise ValueError("You've specified an invalid attack mode.")

            forward_attacked_images = (image_batch[remaining_image_indices] + forward_perts).clamp(0, 1)
            backward_attacked_images = (image_batch[remaining_image_indices] + backward_perts).clamp(0, 1)

            attacked_image_labels = label_batch[remaining_image_indices]
            attacked_image_label_probs = batch_label_probs[remaining_image_indices]

            forward_attack_preds, forward_attack_probs = muffy.get_preds_and_scores(
                model, forward_attacked_images, attacked_image_labels, normalisation_func, apply_softmax=apply_softmax,
                targeted_scores=targeted, subtract_competition=subtract_competition, batch_size=batch_size
            )

            if query_saver:
                candidate_probs_one = torch.stack([forward_attack_probs, attacked_image_label_probs], 0)
                best_probs_one, best_attacks_one = prob_selector(candidate_probs_one)

                # First, we try just one direction (arbitrarily, the forward one), and see whether it reduces the loss.
                # If so, we keep it and don't bother trying the other sign for this direction, saving ourselves a
                #   query. If not, we spend the other query to check the other direction.
                # If the forward direction won, you're done for now:
                required_one = (best_attacks_one == 0).nonzero().flatten()
                # If the original position won, keep going:
                required_two = (best_attacks_one == 1).nonzero().flatten()

                remaining_query_count[required_one] += 1
                remaining_query_count[required_two] += 2
                batch_query_count[remaining_image_indices] = remaining_query_count

                best_attacks = torch.ones(remaining_image_count).long() * 2
                best_attacks[required_one] = 0
                best_probs = best_probs_one

                backward_attack_preds = torch.ones(remaining_image_count).long() * (-1)
                backward_attack_probs = torch.ones(remaining_image_count) * float('nan')
                if required_two.numel() > 0:
                    # required_two contains the indices *within* remaining_image_indices that need further work.
                    # We only run the second query for those. Other predicted class/score values are invalid in the
                    #   opposite direction, and should never be read:
                    backward_attack_preds_req_two, backward_attack_probs_req_two = muffy.get_preds_and_scores(
                        model, backward_attacked_images[required_two], attacked_image_labels[required_two],
                        normalisation_func, apply_softmax=apply_softmax, targeted_scores=targeted,
                        subtract_competition=subtract_competition, batch_size=batch_size
                    )
                    backward_attack_preds[required_two] = backward_attack_preds_req_two
                    backward_attack_probs[required_two] = backward_attack_probs_req_two
                    candidate_probs_two = torch.stack(
                        [backward_attack_probs[required_two], attacked_image_label_probs[required_two]], 0)
                    best_probs_two, best_attacks_two = prob_selector(candidate_probs_two)
                    backward_win_inds = required_two[best_attacks_two == 0]
                    best_attacks[backward_win_inds] = 1
                    best_probs[required_two] = best_probs_two

            else:
                backward_attack_preds, backward_attack_probs = muffy.get_preds_and_scores(
                    model, backward_attacked_images, attacked_image_labels, normalisation_func,
                    apply_softmax=apply_softmax, targeted_scores=targeted, subtract_competition=subtract_competition,
                    batch_size=batch_size
                )
                # We want to assemble the three potential probs, pick a best one, and have an elegant way of keeping
                #   that mapped to the relevant structures to be able to slice from them as needed to write the winners
                #   into place, ready to be iterated on the next time through:
                candidate_probs = torch.stack([forward_attack_probs, backward_attack_probs,
                                               attacked_image_label_probs], 0)
                best_probs, best_attacks = prob_selector(candidate_probs)
                batch_query_count[remaining_image_indices] += 2

            # You might be thinking, "wait, can't you just slice the slice, increment those values, and be done with
            #   it?" Well, I tried, and computer said no.
            remaining_direction_taken_count[best_attacks != 2] += 1
            batch_direction_taken_count[remaining_image_indices] = remaining_direction_taken_count

            # Assemble the additional components of the canonical attacks in COO format:
            forward_attack_batch_row_inds = torch.cat(
                [forward_attack_batch_row_inds,
                 remaining_image_indices[(best_attacks == 0).nonzero().flatten()].flatten()]
            )
            backward_attack_batch_row_inds = torch.cat(
                [backward_attack_batch_row_inds,
                 remaining_image_indices[(best_attacks == 1).nonzero().flatten()].flatten()]
            )
            # Note that the basis index is recorded as basis_id, not attack_basis_inds[basis_id]. The understanding is
            #   that the client will iterate through the indices they supplied, accordingly. It shows you, in a
            #   straightforward way, at which index/iteration a particular component was added, which allows you to
            #   "replay" the progression naturally if desired, after the fact. It allows avoiding the use of redundant
            #   structures if desired, as well.
            # (Apologies for how roundabout the below is. It could've been much simpler.)
            forward_attack_batch_col_inds = torch.cat([forward_attack_batch_col_inds,
                                                       torch.ones(len((best_attacks == 0).nonzero().flatten())).long()
                                                       * basis_id])
            backward_attack_batch_col_inds = torch.cat([backward_attack_batch_col_inds,
                                                       torch.ones(len((best_attacks == 1).nonzero().flatten())).long()
                                                       * basis_id])

            candidate_basis_sums = torch.stack([forward_basis_sum_candidates, backward_basis_sum_candidates,
                                                current_basis_sum_images], 0)
            current_basis_sum_images = candidate_basis_sums[best_attacks, range(len(best_attacks))]
            if return_adv_tensor:
                candidate_pert_imgs = torch.stack([forward_perts, backward_perts, current_adv_pert_images], 0)
                current_adv_pert_images = candidate_pert_imgs[best_attacks, range(len(best_attacks))]

            # Now splat your updated data back into the main container:
            basis_sum_images[remaining_image_indices] = current_basis_sum_images.clone()
            if return_adv_tensor:
                adv_pert_images[remaining_image_indices] = current_adv_pert_images.clone()
            batch_label_probs[remaining_image_indices] = best_probs.clone()

            # And finally, update the remaining image indices by crossing off any completions:
            candidate_preds = torch.stack([forward_attack_preds, backward_attack_preds,
                                           batch_class_preds[remaining_image_indices]], 0)
            batch_class_preds[remaining_image_indices] = candidate_preds[best_attacks, range(len(best_attacks))]
            remaining_image_indices = label_comparator(batch_class_preds, label_batch).nonzero().flatten()

            print(f"\tLast basis ID used: {basis_id}; Images remaining: {len(remaining_image_indices)}")

            basis_id += 1

        canonical_sp_inds = torch.cat(
            [canonical_sp_inds,
             torch.stack([torch.cat([forward_attack_batch_row_inds, backward_attack_batch_row_inds]) + batch_begin_ind,
                          torch.cat([forward_attack_batch_col_inds, backward_attack_batch_col_inds])], dim=0)],
            dim=1)
        # (This is a {-1, 0, 1} sign matrix.)
        canonical_sp_vals = torch.cat([canonical_sp_vals,
                                       torch.ones(len(forward_attack_batch_row_inds)),
                                       -torch.ones(len(backward_attack_batch_row_inds))])

        if return_adv_tensor:
            adv = torch.cat([adv, adv_pert_images], dim=0)
        batch_succs = torch.ones(len(image_batch)).bool()
        batch_succs[remaining_image_indices] = False
        succs = torch.cat([succs, batch_succs], dim=0)
        direction_taken_count = torch.cat([direction_taken_count, batch_direction_taken_count], dim=0)
        query_count = torch.cat([query_count, batch_query_count], dim=0)

    canonical_adv = torch.sparse.FloatTensor(canonical_sp_inds, canonical_sp_vals, canonical_sp_size).coalesce()
    # To regenerate full adversarial perturbations from canonical representations (if the full adversaries weren't
    #   saved out originally), the client can call regenerate_adversarial_perturbations, in mufasa_utils.

    return canonical_adv, succs, query_count, adv


def mufasa():
    parser = configargparse.ArgumentParser(
        description="A version of SimBA which allows for the use of your own basis definition, as well as offering "
                    "some other expanded functionality.",
        add_help=False
    )

    required_named_arguments = parser.add_argument_group("required named arguments")
    required_named_arguments.add_argument(
        '--model', type=str, required=True,
        help="Either the name of the pretrained torchvision model to use (attempted first), or the local path to a "
             "saved model. Note that the model's definition will need to be available for the latter option to work, "
             "which the user will have to take care of themselves."
    )
    required_named_arguments.add_argument(
        '--dataset', type=str, required=True,
        choices=['mnist', 'cifar10_train', 'cifar10_test', 'imagenet_val', 'image_folder'],
        help="The name of the dataset to use. ('image_folder' uses ImageFolder to load an arbitrary set defined by "
             "collections of subfolders: see docs.)"
    )
    required_named_arguments.add_argument(
        '--data_dir', type=str, required=True,
        help="The master folder of the dataset."
    )
    required_named_arguments.add_argument(
        '--data_transform', type=str, required=True,
        choices=['cifar_common_32', 'imagenet_common_224', 'imagenet_inception_299'],
        help="The name specifying the data transformation to use on loading the given dataset. This will also "
             "necessarily set the working image size (per-side dimension of a square image). If your desired transform "
             "isn't available, you'll have to implement it and add it as a choice here. You must ensure it is "
             "appropriate for your chosen model and dataset."
    )
    required_named_arguments.add_argument(
        '--data_normalisation', type=str, required=True,
        choices=['mnist_common', 'mnist_basic', 'cifar_common', 'cifar_basic', 'imagenet_common', 'none'],
        help="The name specifying the normalisation applied to the input image before being evaluated by the net. This "
             "will often involve mean and std normalisation, but not necessarily. If your desired normalisation isn't "
             "available, you'll have to implement it and add it as a choice here. You must ensure it is appropriate "
             "for your chosen model and dataset."
    )
    required_named_arguments.add_argument(
        '--num_runs', type=int, required=True,
        help="The number of images to sample and perturb."
    )
    required_named_arguments.add_argument(
        '--attack_basis', type=str, required=True,
        choices=['pixel', 'dct', 'enlo'],
        help="The basis vectors representing the bank of attack directions. The 'enlo' option means that the user will "
             "specify the basis themselves, using the 'basis_path' option."
    )
    required_named_arguments.add_argument(
        '--order', type=str, required=True,
        choices=['straight', 'random'],
        help='Scheme for ordering the (potentially random) attack coordinate selection.'
    )

    optional_arguments = parser.add_argument_group("optional arguments")
    # We switched help off in order to get our required arguments before our optional ones. Now we just add it back:
    optional_arguments.add_argument(
        '-h', '--help', action='help', default=configargparse.SUPPRESS,
        help="show this help message and exit"
    )
    optional_arguments.add_argument(
        '--config_file', is_config_file=True,
        help="Optional file from which to read parameter values. In the case of multiple specifications, the override "
             "order is (command line) > (environment vars) > (config file) > (defaults), as in the ConfigArgParse "
             "docs. See the docs for the valid config file format options."
    )
    optional_arguments.add_argument(
        '--targeted', action='store_true',
        help="If specified, perform a targeted attack (else untargeted). (default: %(default)s)"
    )
    optional_arguments.add_argument(
        '--epsilon', type=float, default=0.2,
        help="The magnitude of the perturbation along the chosen basis vector in a single iteration. "
             "(default: %(default)f)"
    )
    optional_arguments.add_argument(
        '--norm_bound', type=float, default=float('inf'),
        help="The (l2) bound on the norm of the attack, enforced by adding a projection step to the standard SimBA "
             "search. If not specified, the bound is infinite, i.e. there is no bound. (default: %(default)f)"
    )
    optional_arguments.add_argument(
        '--num_iters', type=int, default=0,
        help="The maximum number of attack iterations. 0 is interpreted as 'unlimited', i.e. the method will iterate "
             "indefinitely until all attacks are successful. (default: %(default)d)"
    )
    optional_arguments.add_argument(
        '--freq_domain_width', type=int, default=14,
        help="Width of the square grid representing 2D frequency space in the 'dct' attack option. "
             "(default: %(default)d)"
    )
    optional_arguments.add_argument(
        '--apply_softmax', action='store_true',
        help="If specified, class-score logits are passed through softmax (i.e. are 'confidences') before being fed to "
             "energy function, otherwise not. (default: %(default)s)"
    )
    optional_arguments.add_argument(
        '--subtract_competition', action='store_true',
        help="If specified, the energy function subtracts the score of the most competitive class (according to "
             "targeting context), otherwise not. (default: %(default)s)"
    )
    optional_arguments.add_argument(
        '--attack_mode', type=str, default='standard_simba',
        choices=['standard_simba', 'linf_rescale', 'linf_signs'],
        help="This option defines how epsilon is interpreted, and how attacks are scaled and/or applied. "
             "standard_simba: epsilon is the step length, as usual; note the fact that the norm_bound parameter bounds "
             "the total magnitude of the accumulated candidate perturbation in a PGD-esque manner. "
             "linf_rescale: epsilon specifies an l-inf bound on the total attack; the perturbation is rescaled s.t. "
             "its largest-magnitude (pixel) coordinate has magnitude epsilon. "
             "linf_signs: epsilon has the same interpretation as linf_rescale, but *all* coordinates are rescaled to "
             "have this magnitude, i.e., only their signs are used."
    )

    optional_arguments.add_argument(
        '--query_saver', action='store_true',
        help="If specified, uses the reference SimBA approach of trying one forward-difference measurement in each "
             "iteration and greedily choosing the corresponding sign if the loss decreases, before resorting to "
             "checking the effect of the opposite direction. This will sometimes save one query in a given iteration. "
             "Otherwise, both directions are checked in each iteration, for a fixed cost of two queries. (default: "
             "%(default)s)"
    )
    optional_arguments.add_argument(
        '--batch_size', type=int, default=50,
        help="The batch size for parallel runs. (default: %(default)d)"
    )
    optional_arguments.add_argument(
        '--sampled_image_dir', type=str, default='image_cache',
        help="Directory for caching sampled images. (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--correct_samples_only', action='store_true',
        help="If specified, the attack test set will consist only of images that were originally correctly classified "
             "by the network, otherwise not. The cache name will reference the net if and only if this option is "
             "specified. (default: %(default)s)"
    )
    optional_arguments.add_argument(
        '--save_suffix', type=str, default='',
        help="(Optional) suffix appended to the save file name. (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--result_dir', type=str, default='mufasa_results',
        help="Directory for saving results. (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--return_adv_tensor', action='store_true',
        help="If specified, return the image-space adversary tensor, equal in size to the input image tensor (else "
             "only the sparse canonical representation is returned). (default: %(default)s)"
    )
    optional_arguments.add_argument(
        '--basis_path', type=str, default='',
        help="If the attack basis is being provided from disc (as in the 'enlo' option), this supplies the path (incl. "
             "filename) to the saved basis file, which will be loaded with torch.load. Currently assumes that load "
             "will return two values, (u, s), with the first one representing the basis matrix and the second a set of "
             "importance scores (e.g. left singular vectors and singular values, respectively). "
             "(default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--save_prefix', type=str, default='mufasa_',
        help="The (optional) string to prepend to the name of saved files. Note that this is passed to the net and "
             "data retrieval function as well, which uses it in finding and creating image cache files. Note also that "
             "if you want an underscore to separate it from the rest of the filename, you must include one. "
             "(default: '%(default)s')"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sampled_image_dir = Path(args.sampled_image_dir)
    result_dir = Path(args.result_dir)
    basis_path = Path(args.basis_path)

    result_dir.mkdir(parents=True, exist_ok=True)

    model, images, labels, num_classes, image_size, normaliser, mean, std, cache = muffy.retrieve_net_and_data_samples(
        sampled_image_dir, args.save_prefix, args.dataset, args.data_transform, args.model, args.data_normalisation,
        args.num_runs, data_dir, args.correct_samples_only, args.batch_size)

    if args.attack_basis == 'pixel':
        attack_to_image_space_transform = pixel_basis_meta_fun(image_size)
        attack_space_dim = torch.tensor(image_size).prod()
    elif args.attack_basis == 'dct':
        attack_size = torch.Size((image_size[0], args.freq_domain_width, args.freq_domain_width))
        attack_to_image_space_transform = dct_lf_basis_meta_fun(image_size, attack_size)
        attack_space_dim = torch.tensor(attack_size).prod()
    elif args.attack_basis == 'enlo':
        attack_to_image_space_transform, attack_space_dim = enlo_basis_meta_fun(image_size, basis_path)
    else:
        raise ValueError("You've specified an invalid attack basis.")

    # The labels the attacker requires are the target labels in the targeted case (go towards those), else the
    #   ground-truth labels (go away from those).
    if args.targeted:
        attack_labels = labels.clone()
        while attack_labels.eq(labels).any():
            still_equal_indices = attack_labels.eq(labels).nonzero().flatten()
            attack_labels[still_equal_indices] = torch.randint(num_classes, (len(still_equal_indices),))
    else:
        attack_labels = labels.clone()  # Making a copy just in case the attacker wants to manipulate the labels.

    num_images = len(images)
    num_batches = math.ceil(num_images / args.batch_size)

    if args.return_adv_tensor:
        adv = torch.Tensor()
    else:
        adv = None
    succs = torch.BoolTensor()
    queries = torch.LongTensor()
    
    if args.num_iters > 0:
        attacks_size = min(attack_space_dim, args.num_iters)
    else:
        attacks_size = attack_space_dim
    
    canonical_adv = torch.sparse.FloatTensor(0, attacks_size)
    active_attack_basis_indices = torch.sparse.LongTensor(0, attacks_size)

    for b in range(num_batches):

        batch_begin_ind, batch_end_ind = (b * args.batch_size), min((b + 1) * args.batch_size, num_images)
        image_batch, attack_label_batch = (images[batch_begin_ind:batch_end_ind],
                                           attack_labels[batch_begin_ind:batch_end_ind])

        if args.order == 'straight':
            attack_basis_indices = torch.arange(attack_space_dim).unsqueeze(0).repeat(len(image_batch), 1)
        elif args.order == 'random':
            attack_basis_indices = torch.LongTensor(len(image_batch), attack_space_dim)
            for r in range(len(attack_basis_indices)):
                attack_basis_indices[r, :] = torch.randperm(attack_space_dim)
        else:
            raise ValueError("I don't support that order yet. :-D")
        if 0 < args.num_iters < attack_space_dim:
            attack_basis_indices = attack_basis_indices[:, :args.num_iters]

        batch_canonical_adv, batch_succs, batch_queries, batch_adv = simba_core(
            model, normaliser, image_batch, attack_label_batch, attack_to_image_space_transform, attack_basis_indices,
            args.targeted, args.epsilon, args.norm_bound, apply_softmax=args.apply_softmax,
            subtract_competition=args.subtract_competition, attack_mode=args.attack_mode, query_saver=args.query_saver,
            batch_size=args.batch_size, return_adv_tensor=args.return_adv_tensor
        )

        if adv is not None:
            adv = torch.cat([adv, batch_adv], dim=0)
        succs = torch.cat([succs, batch_succs], dim=0)
        queries = torch.cat([queries, batch_queries], dim=0)
        canonical_adv = torch.cat([canonical_adv, batch_canonical_adv], dim=0)
        active_attack_basis_indices = torch.cat([active_attack_basis_indices,
                                                 attack_basis_indices.sparse_mask(batch_canonical_adv)], dim=0)

    prefix = args.save_prefix + args.attack_basis
    if args.attack_basis == 'dct':
        prefix += '_' + str(args.freq_domain_width)
    if args.targeted:
        prefix += '_targeted'
    save_filename = result_dir / (f"{prefix}_{clean_model_name(args.model)}_{args.dataset}_{args.num_runs}_"
                                  f"{args.num_iters}_{args.epsilon:.4f}_{args.norm_bound:.4f}_{args.order}"
                                  f"{args.save_suffix}.pt")

    out_dict = {'canonical_adv': canonical_adv, 'succs': succs, 'queries': queries, 'adv': adv,
                'active_attack_basis_indices': active_attack_basis_indices, 'attack_labels': attack_labels,
                'input_args': args}

    if args.attack_basis == 'enlo':
        out_dict['basis_path'] = basis_path.name

    torch.save(out_dict, save_filename)


if __name__ == '__main__':
    mufasa()
