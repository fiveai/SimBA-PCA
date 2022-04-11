# WARNING: Note that you should never input any images to this method that you will eventually test a derived black-
#   box attack on, as that constitutes leakage. You can do this as a *clearly labelled reference experiment* if you
#   like, but you must slap a special warning label on any such runs. (The normal/safe thing to do would be to generate
#   directions on training splits and extract basis vectors from those, then hand those to Mufasa to use to attack the
#   corresponding val sets. You can try this using the exact same net, the same architecture but different weights, and
#   a completely different net, to evaluate different attack scenarios. You can likewise study the extent of transfer
#   of bases across different datasets, with appropriate resizing if necessary.)

# This is nothing but a glorified loop that spits out gradients/adversarial examples sampled at different points in
#   image space for a given net. This is typically done using a dataset of real images to be attacked, but in principle,
#   you might want to sample image-space points differently, and may not even require a "real" dataset to do it (because
#   e.g. you're just sampling image space in some random fashion).

# The main purpose of this is to supply directions to be assembled and decomposed by dominant_directions.py, which
#   itself supplies basis directions to be used in more sophisticated/universal attacks, particularly certain black-box
#   settings.

# This makes extensive use of Foolbox (3.1.1).

# Note that I'll be using "adversarial example" or "adversarial direction" loosely unless and until I say otherwise. The
#   point here is to represent directions of network sensitivity, not necessarily to provide successful or otherwise
#   desirable adversarial attacks per se. E.g. one might return an unscaled gradient because it serves the ultimate
#   purpose. An effective and arguably good way of doing this is by just computing adversarial examples in the usual
#   sense, and options of that sort will certain comprise a large part of what actually happens here. But there may
#   options available that do not quite fit that bill, so the outputs here shouldn't be taken as anything other than the
#   input to a method that understands what they represent. This is ultimately a utility within a larger project.
# Relating to the above point, note that the lengths of adversarial perturbations can be inversely related to the
#   effectiveness of the attack direction (in the case of any attack seeking to minimise the perturbation norm, e.g.
#   anything in the DeepFool/Carlini-Wagner family). Now, we experimented with different variants behind the scenes of
#   With Friends Like These, and we curiously (at least for simple CIFAR nets) never found any meaningful difference
#   between using gradients, normalised gradients, and normalised gradients divided by their original magnitudes!
#   We never found a satisfactory explanation. The inconclusiveness is all the more reason to stay conscious of which
#   option one is selecting here, and what the implications of that choice might be when the next processing steps take
#   place. A wide menu of options are offered, at least for the time being, to reflect the uncertainty around this
#   point: the user should take care, keep an open mind, and stay vigilant.

import configargparse
import json
import math
from pathlib import Path

import torch
import torchvision.utils as tv_utils

import foolbox as fb

import src.mufasa_utils as muffy
from src.str_parsing import clean_model_name


def generate_adversaries():

    parser = configargparse.ArgumentParser(
        description="Method for generating a collection of samples of saliency directions (which may be adversarial "
                    "examples in the usual sense or e.g. just raw gradients) for a given network. Typically, this will "
                    "be done at image-space points represented by real images from a given dataset, but note that the "
                    "user can supply whatever image-space data they like, including e.g. randomly sampled points.",
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
        choices=['mnist', 'cifar10_train', 'cifar10_test', 'imagenet_train', 'imagenet_val', 'image_folder'],
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
        choices=['mnist_common', 'cifar_common', 'cifar_basic', 'imagenet_common', 'none'],
        help="The name specifying the normalisation applied to the input image before being evaluated by the net. This "
             "will often involve mean and std normalisation, but not necessarily. If your desired normalisation isn't "
             "available, you'll have to implement it and add it as a choice here. You must ensure it is appropriate "
             "for your chosen model and dataset."
    )
    required_named_arguments.add_argument(
        '--adversary_type', type=str, required=True,
        choices=['normalised_data', 'raw_gradients', 'FGM', 'FGSM', 'BIML2', 'BIMLinf', 'PGDL2', 'PGDLinf', 'CWL2',
                 'DeepFoolL2', 'DeepFoolLinf'],
        help="The choice of adversarial attack or other related information (e.g. grad w.r.t. image) to be computed "
             "over the sample set. Adversaries will typically be implemented as Foolbox calls."
    )
    required_named_arguments.add_argument(
        '--sample_count', type=int, required=True,
        help="The number of samples/adversaries to be computed: each will be output as a separate file"
    )

    optional_arguments = parser.add_argument_group("optional arguments")

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
        '--sampled_image_dir', type=str, default='image_cache',
        help="Directory for caching sampled images. (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--correct_samples_only', action='store_true',
        help="If specified, adversary samples will be based only on images that were originally correctly classified "
             "by the network, otherwise not. The cache name will reference the net if and only if this option is "
             "specified. (default: %(default)s)"
    )
    optional_arguments.add_argument(
        '--epsilon', type=float, default=float('inf'),
        help="The epsilon value to be passed to any Foolbox attack that might have been chosen. In 3.1.1, this is the "
             "norm bound, *in input space* (with no e.g. sqrt(D) normalisation). Take care to understand how this "
             "interacts with your chosen attack. For some methods (e.g. DeepFool), it is completely appropriate not to "
             "provide a norm bound; for others, it's essential. (default: %(default)f)"
    )
    # You can add, e.g., for Carlini-Wagner, a line in the input file like this:
    # --attack_options={"steps": 100, "initial_const": 0.1}
    optional_arguments.add_argument(
        '--attack_options', type=json.loads, default=dict(),
        help="A dictionary of parameters to be unpacked as the optional arguments to whatever attack/direction "
             "computation you've chosen using 'adversary_type'. Keys are parameter names and values are their values. "
             "Note that you may have to be careful with how you use inverted commas in defining the dictionary, "
             "depending on whether it's on the command line or in a file. (default: %(default)s)"
    )
    optional_arguments.add_argument(
        '--result_dir', type=str, default='results',
        help="Directory for saving results. (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--save_suffix', type=str, default='',
        help="Suffix appended to the save file name. (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--batch_size', type=int, default=25,
        help="The batch size for parallel runs. (default: %(default)d)"
    )
    optional_arguments.add_argument(
        '--save_prefix', type=str, default='generate_adversaries_',
        help="The (optional) string to prepend to the name of saved files. Note that this is passed to the net and "
             "data retrieval function as well, which uses it in finding and creating image cache files. Note also that "
             "if you want an underscore to separate it from the rest of the filename, you must include one. "
             "(default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--save_images', action='store_true',
        help="If specified, also saved attacked images as png, for inspections. (default: %(default)s)"
    )

    args = parser.parse_args()

    sampled_image_dir = Path(args.sampled_image_dir)
    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)

    model, images, labels, num_classes, image_size, normaliser, mean, std, cache = muffy.retrieve_net_and_data_samples(
        sampled_image_dir, args.save_prefix, args.dataset, args.data_transform, args.model, args.data_normalisation,
        args.sample_count, data_dir, args.correct_samples_only, args.batch_size)

    preprocessing = dict(mean=mean, std=std, axis=-3)
    bounds = (0, 1)  # We're always working under the assumption of data valued within [0, 1].
    # You no longer need to tell Foolbox the number of classes. (I'm assuming v3+).
    foolbox_model = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    if args.adversary_type == 'FGM':
        attack = fb.attacks.FGM(**args.attack_options)  # This should simply be a rescaled gradient (of length epsilon).
    elif args.adversary_type == 'FGSM':
        attack = fb.attacks.FGSM(**args.attack_options)
    elif args.adversary_type == 'BIML2':
        attack = fb.attacks.L2BasicIterativeAttack(**args.attack_options)
    elif args.adversary_type == 'BIMLinf':
        attack = fb.attacks.LinfBasicIterativeAttack(**args.attack_options)
    elif args.adversary_type == 'PGDL2':  # Given the options provided for the two attacks, and having not been through
        # the source, I am not clear what, if anything, separates "PGD" from "BIM" in this package. In earlier versions
        # of the docs, it even says of PGD, "When used without a random start, this attack is also known as Basic
        # Iterative Method (BIM) or FGSM^k." The random start, so far as I can tell from the source papers, is the only
        # real "difference", and the current BIM implementation offers that option, so ¯\_(ツ)_/¯. Btw, in Foolbox, they
        # have pretty different default step sizes and iteration counts for the L2 versions, making BIM the "coarser"
        # one by default.
        attack = fb.attacks.L2ProjectedGradientDescentAttack(**args.attack_options)
    elif args.adversary_type == 'PGDLinf':  # See above.
        attack = fb.attacks.LinfProjectedGradientDescentAttack(**args.attack_options)
    elif args.adversary_type == 'DeepFoolL2':
        attack = fb.attacks.L2DeepFoolAttack(**args.attack_options)
    elif args.adversary_type == 'DeepFoolLinf':
        attack = fb.attacks.LinfDeepFoolAttack(**args.attack_options)
    elif args.adversary_type == 'CWL2':
        attack = fb.attacks.L2CarliniWagnerAttack(**args.attack_options)  # Note that you will almost certainly want
        # to tune the options of this method if you actually use it, unless you want to wait forever. Seek guidance.

    # Note: You have to make choices around whether you think it's better to clip the perturbation or not, for our
    #   intended purposes. I'd say it's better to use the raw perturbation for analysis, but save the clipped image for
    #   viewing. This should *very* seldom make any practical difference for norm-optimal perturbations, but you might
    #   deliberately be specifying options with much larger perturbations (though in those cases, I'd say how you should
    #   best visualise the result becomes more subjective anyway). e.g. if what you want is a raw gradient, you might
    #   actually want to shift and normalise it to fit within the image gamut to see the "pattern", under the advice
    #   that that has been done to what you're looking at.

    clean_model_str = clean_model_name(args.model)
    adv_pert_save_folder = result_dir / (
        f"{args.save_prefix}raw_pert_{clean_model_str}_{args.dataset}_{args.adversary_type}_{args.sample_count}_"
        f"{args.epsilon:.4f}"
    )
    adv_pert_save_folder.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        adv_image_save_folder = result_dir / (
            f"{args.save_prefix}clipped_im_{clean_model_str}_{args.dataset}_{args.adversary_type}_{args.sample_count}_"
            f"{args.epsilon:.4f}"
        )
        adv_image_save_folder.mkdir(parents=True, exist_ok=True)

    num_batches = math.ceil(images.size(0) / args.batch_size)
    for b in range(num_batches):
        batch_range_start, batch_range_end = (b * args.batch_size), min((b + 1) * args.batch_size, images.size(0))
        image_batch = images[batch_range_start:batch_range_end]
        label_batch = labels[batch_range_start:batch_range_end]
        if args.adversary_type == 'normalised_data':  # This isn't really an attack: the "perturbations" are just
            # normalised input data. This is analogous to running PCA on the data. We serve the results like so:
            perts = normaliser(image_batch)
            clipped = image_batch
            is_adv = torch.zeros(len(image_batch)).bool()  # None of these are adversaries.
        elif args.adversary_type == 'raw_gradients':
            # The class score to be differentiated is chosen at random, per image, in this option.
            target_classes = torch.randint(num_classes, label_batch.size())
            input_images = normaliser(image_batch).detach().clone().cuda().requires_grad_()
            f_input_images = model(input_images)
            perts = torch.zeros_like(input_images)  # Pre-allocation, before the gradient loop.
            # Note that it's goofy that that I'm computing the gradient w.r.t. the entire batch when, of course, only
            #   one image is actually relevant to each value (and so the gradient will be zero elsewhere), but, w/e.
            #   Good enough. One forward pass on the batch this way, with less code.
            for i, tc in enumerate(target_classes):
                model.zero_grad()
                f_input_images[i, tc].backward(retain_graph=True)
                perts[i] = input_images.grad[i].detach().clone()
            clipped = image_batch  # Not adding a gradient whose magnitude might be completely inappropriate: just
            # returning dummy images for the adversaries, as with the normalised data option.
            is_adv = torch.zeros(len(image_batch)).bool()  # None of these are adversaries.
        else:  # You've chosen a Foolbox attack.
            raw, clipped, is_adv = attack(foolbox_model, image_batch.cuda(), label_batch.cuda(), epsilons=args.epsilon)
            # These may be in different places, so let's just make sure they're both on the CPU now:
            perts = raw.cpu() - image_batch.cpu()

        for i, p in enumerate(perts):
            adv_pert_save_filename = f"ind{(b*args.batch_size)+i:05d}_adv{is_adv[i]}{args.save_suffix}.pt"
            torch.save(p.clone(), adv_pert_save_folder / adv_pert_save_filename)
            if args.save_images:
                adv_image_save_filename = f"ind{(b*args.batch_size)+i:05d}_adv{is_adv[i]}{args.save_suffix}.png"
                tv_utils.save_image(clipped[i], adv_image_save_folder / adv_image_save_filename)


if __name__ == '__main__':
    generate_adversaries()
