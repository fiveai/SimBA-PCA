# This reads in the outputs from Mufasa and, given the original net (including associated pipeline functions, e.g.
#   preprocessing), applies the attacks, checks that they do indeed work, and, if requested, dumps out a folder of
#   individual attacked image files for inspection. That is, it simulates a client receiving Mufasa's output. If all
#   looks well here, you can be pretty confident that the method has basically done what you think it has.

# Note that for this to work, this method must have access to the same image cache (sample set) for which the results
#   were generated. Note that image caches are created through random sampling of a given dataset based on a requested
#   sample count, and the file named based on those parameters and various others (e.g. specifying the net to be
#   attacked, on which the samples are as of this writing tested for correct classification before inclusion in the
#   set). This means that it is assumed that the actual cache file will not be deleted and (randomly) recreated under
#   the same set of settings unless it is understood that to do so renders all existing results stale and unusable.
# Basically, you shouldn't ever be deleting image cache files unless you are also deleting all corresponding results
#   along with them.

import configargparse
import math
from pathlib import Path

import torch
import torchvision.utils as tv_utils

import mufasa  # This is done to get the basis generator functions. These could/should possibly be in the utility file?
import src.mufasa_utils as muffy


def check_results():
    parser = configargparse.ArgumentParser(
        description='Method that reads in results from mufasa.py, along with specification of the network the results '
                    'correspond to, and applies them to check correctness and generate visualisations.',
        add_help=False
    )

    required_named_arguments = parser.add_argument_group('required named arguments')
    required_named_arguments.add_argument(
        '--input_file', type=str, required=True,
        help='The name of the mufasa result file being applied/checked, including path if necessary.'
    )

    optional_arguments = parser.add_argument_group('optional named arguments')
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
        '--image_save_ext', type=str, default='.png',
        help="The file format in which you want your perturbed images saved (if requested). (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--pert_save_ext', type=str, default='.pt',
        help="The file format in which you want your (individual) perturbations saved (if requested). "
             "(default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--result_dir', type=str, default='check_results_out',
        help="Directory for saving results. (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--dump_images', action='store_true',
        help="If specified, output a folder of adversarial image files, else, don't. (default: %(default)s)"
    )
    optional_arguments.add_argument(
        '--dump_perturbations', action='store_true',
        help="If specified, output a folder of individual adversarial perturbation files, else, don't. These can be "
             "used directly by dominant directions. (default: %(default)s)"
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    input_data = torch.load(input_file)

    canonical_adv = input_data['canonical_adv']
    succs = input_data['succs']
    adv = input_data['adv']
    attack_basis_indices = input_data['active_attack_basis_indices']
    attack_labels = input_data['attack_labels']
    mufasa_input_args = input_data['input_args']

    # Note that if there is anything other than a cache hit here, something has gone wrong and this should fail.
    #   i.e. the cache has gone missing. Ideally, that case would be caught and the failure made explicit, but there's
    #   no provision for this at present.
    model, images, labels, num_classes, image_size, normaliser, _, _, cache = muffy.retrieve_net_and_data_samples(
        Path(mufasa_input_args.sampled_image_dir), mufasa_input_args.save_prefix, mufasa_input_args.dataset,
        mufasa_input_args.data_transform, mufasa_input_args.model, mufasa_input_args.data_normalisation,
        mufasa_input_args.num_runs, Path(mufasa_input_args.data_dir), mufasa_input_args.correct_samples_only,
        mufasa_input_args.batch_size)

    num_images = len(images)
    num_batches = math.ceil(num_images / mufasa_input_args.batch_size)

    if adv is None:  # We must "replay" the attacks from our efficient storage structures to construct them as images
        # first. Otherwise, we skip this construction and proceed directly.
        if mufasa_input_args.attack_basis == 'pixel':
            attack_to_image_space_transform = mufasa.pixel_basis_meta_fun(image_size)
        elif mufasa_input_args.attack_basis == 'dct':
            attack_size = torch.Size(
                (image_size[0], mufasa_input_args.freq_domain_width, mufasa_input_args.freq_domain_width)
            )
            attack_to_image_space_transform = mufasa.dct_lf_basis_meta_fun(image_size, attack_size)
        elif mufasa_input_args.attack_basis == 'enlo':
            attack_to_image_space_transform, attack_space_dim = mufasa.enlo_basis_meta_fun(
                image_size, mufasa_input_args.basis_path
            )
        else:
            raise ValueError("You've specified an invalid attack basis.")

        canonical_adv_scipy = muffy.convert_sparse_tensor_to_scipy(canonical_adv)
        attack_basis_indices_scipy = muffy.convert_sparse_tensor_to_scipy(attack_basis_indices)

    for b in range(num_batches):
        batch_begin_ind, batch_end_ind = (b * mufasa_input_args.batch_size,
                                          min((b + 1) * mufasa_input_args.batch_size, num_images))
        image_batch = images[batch_begin_ind:batch_end_ind]
        attack_label_batch = attack_labels[batch_begin_ind:batch_end_ind]
        succs_batch = succs[batch_begin_ind:batch_end_ind]

        if adv is None:
            # You have to be careful with certain sparse matrix details, e.g. SciPy's questionable handling of explicit
            #   zeros. (We won't even talk about the current state of PyTorch in this regard.) The below is one way of
            #   doing it that should be fine.
            canonical_adv_batch = torch.tensor(
                canonical_adv_scipy.tocsr()[batch_begin_ind:batch_end_ind, :].todense()
            )
            attack_basis_indices_batch = torch.tensor(
                attack_basis_indices_scipy.tocsr()[batch_begin_ind:batch_end_ind, :].todense()
            )
            attack_image_batch = muffy.regenerate_adversarial_perturbations(
                canonical_adv_batch, attack_basis_indices_batch, attack_to_image_space_transform,
                mufasa_input_args.epsilon, mufasa_input_args.norm_bound, mufasa_input_args.attack_mode
            )
        else:
            attack_image_batch = adv[batch_begin_ind:batch_end_ind]

        attacked_image_batch = (image_batch + attack_image_batch).clamp(0, 1)
        # Note that when we only want preds, many of the currently mandatory arguments are dummies.
        preds, _ = muffy.get_preds_and_scores(
            model, attacked_image_batch, attack_label_batch, normaliser, apply_softmax=False, targeted_scores=False,
            subtract_competition=False, batch_size=25
        )

        # Come face to face with the following dark fact:
        # Net evaluation results in PyTorch are not guaranteed not to depend on the batch size used.
        # If you don't believe me, see the following:
        #   https://discuss.pytorch.org/t/slightly-different-results-for-different-batch-sizes/43458
        # Why am I telling you this? Because the below code, in which the results (i.e. the adversariality of adv)
        #   are checked, can fail for some batch sizes, and succeed for others. Remember that we in the adversarial
        #   attack business are frequently creating images that lie very near to decision boundaries, so this is
        #   not a hypothetical point.
        # If you are getting mysterious "failures", you can try changing the batch size, or increasing the magnitude
        #   of adv ever so slightly (e.g. multiplying the tensor by 1.001). But if you were going to hand these
        #   attacks out to some client, you would probably want to use the trick of effectively amplifying epsilon
        #   a tiny bit to more or less take care of the randomness (for "does this change the label as advertised"
        #   intents and purposes) once and for all.
        # Even then, the xor check below can/will still fail when you amplify perturbations, because some perturbations
        #   that *didn't* work before *will*. You can easily confirm this for such runs by simply changing the below
        #   xors to ors and watching the check change from "fail" to "pass".
        # I believe that the bottom line is, so long as you're using a GPU to compute the adversaries, you're stuck
        #   with the possibility that re-evaluating them will just barely yield the opposite of the result (label
        #   did or did not flip) that you originally observed.
        if mufasa_input_args.targeted:
            result_confirmed = (preds == attack_label_batch) ^ ~succs_batch
        else:
            result_confirmed = (preds != attack_label_batch) ^ ~succs_batch

        if mufasa_input_args.attack_mode in ['linf_rescale', 'linf_signs']:
            attack_norms = attack_image_batch.abs().amax(dim=(1, 2, 3,))
            norms_within_bound = attack_norms <= mufasa_input_args.epsilon
        elif mufasa_input_args.attack_mode in ['standard_simba']:
            tolerance = 1e-2  # Because numerics, you can exceed the norm ball by a tiny amount. I'm allowing it.
            attack_norms = attack_image_batch.norm(dim=(1, 2, 3,), p=2)
            norms_within_bound = attack_norms <= mufasa_input_args.norm_bound + tolerance
        else:  # We're not enforcing a check on any other options yet.
            pass

        print(f"It's {result_confirmed.all()} that all of your results in batch {b} are confirmed correct! "
              f"{len(result_confirmed.nonzero())} out of {len(result_confirmed)} examples match.")

        print(f"It's {norms_within_bound.all()} that all of your results in batch {b} are within their norm bound! "
              f"{len(norms_within_bound.nonzero())} out of {len(norms_within_bound)} examples are within the bound.")

        if args.dump_images:
            adv_save_folder = Path(args.result_dir) / (input_file.name + '_images')
            adv_save_folder.mkdir(parents=True, exist_ok=True)
            for ind, im in enumerate(attacked_image_batch):
                full_ind = batch_begin_ind + ind
                adv_save_filename = f'ind{full_ind:08d}_adv{succs[full_ind]}{args.image_save_ext}'
                tv_utils.save_image(im, adv_save_folder / adv_save_filename)

        if args.dump_perturbations:
            pert_save_folder = Path(args.result_dir) / (input_file.name + '_perturbations')
            pert_save_folder.mkdir(parents=True, exist_ok=True)
            for ind, pert in enumerate(attack_image_batch):
                full_ind = batch_begin_ind + ind
                pert_save_filename = f'ind{full_ind:08d}_adv{succs[full_ind]}{args.pert_save_ext}'
                torch.save(pert.clone(), pert_save_folder / pert_save_filename)


if __name__ == '__main__':
    check_results()
