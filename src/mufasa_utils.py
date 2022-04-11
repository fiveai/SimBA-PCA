# This is for sharing pieces of code that are reusable within the context of the project, i.e. between the different
#   main methods that handle the respective jobs comprising the project as a whole.

import os
import math
import random
import importlib
from pathlib import Path
from collections import deque

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from src.str_parsing import clean_model_name
from src.dataset_statistics import MNIST_BASIC_MEAN, MNIST_BASIC_STD
from src.dataset_statistics import CIFAR_COMMON_MEAN, CIFAR_COMMON_STD, CIFAR_BASIC_MEAN, CIFAR_BASIC_STD
from src.dataset_statistics import IMAGENET_COMMON_MEAN, IMAGENET_COMMON_STD
from src.sample_lists import get_model_specific_indices

import scipy.sparse as scisp
import matplotlib.pyplot as plt


# Note that the supplied labels work differently than in the original SimBA implementation, thus avoiding code
#   duplication.
# The predictions returned are the top class IDs, the probabilities are those corresponding to the supplied label.
# Note that these will in general represent different classes. But whether this is being used in the context of a
#   targeted attack or an untargeted attack, this is the behaviour you want (here).
def get_preds_and_scores(model, inputs, labels, normalisation_func, apply_softmax, targeted_scores,
                         subtract_competition, batch_size=25, return_cpu=True):
    num_batches = math.ceil(inputs.size(0) / batch_size)
    all_preds, all_probs = torch.Tensor().long(), torch.Tensor()
    for i in range(num_batches):
        batch_range_start, batch_range_end = (i * batch_size), min((i + 1) * batch_size, inputs.size(0))
        normalised_input = normalisation_func(inputs[batch_range_start:batch_range_end])
        batch_labels = labels[batch_range_start:batch_range_end]
        with torch.no_grad():
            output = model.cuda()(normalised_input.cuda())
            if apply_softmax:
                output = torch.nn.Softmax()(output)
        top_probs, pred = output.max(1)
        prob = output[range(len(output)), batch_labels]
        if subtract_competition:
            # There is no "the kth largest element/value" function. You can at present choose between "the k largest
            #   elements" and "the kth smallest element".
            # "kthvalue(output.size(1)-1, 1)" means "give me the 2nd-to-least-smallest, i.e. 2nd-biggest, element."
            pred_cpu = pred.cpu()
            if targeted_scores:
                competitor_probs = top_probs.clone()  # The class to suppress is the top one...
                just_succeeded = (pred_cpu == batch_labels)
                if just_succeeded.nonzero().numel() > 0:
                    competitor_probs[just_succeeded], _ = output[just_succeeded].kthvalue(output.size(1)-1, 1)
                # ^ ... except when you've just succeeded, when the target class is in 1st place and we want 2nd.
            else:
                competitor_probs, _ = output.kthvalue(output.size(1)-1, 1)  # The class to boost is the 2nd-place one...
                competitor_probs[pred_cpu != batch_labels] = top_probs.clone()[pred_cpu != batch_labels]
                # ^... except when you've just succeeded: the source class has slipped out of 1st, and we want 1st.
            prob -= competitor_probs
        if return_cpu:
            prob = prob.cpu()
            pred = pred.cpu()
        all_probs = torch.cat((all_probs, prob))
        all_preds = torch.cat((all_preds, pred))
    return all_preds, all_probs


# simba_core takes a normalisation function along with the network, since in PyTorch, normalisation isn't part of the
#   network itself. Here, it's being kept general, viewed as some function that will be applied to the data. This
#   generator gives you such a function in exchange for a string specifying the name of the function you want. If you
#   want one that hasn't been implemented, implement it.
# Now, as I say, I chose to return a function, because, it's cleaner in many contexts (including in my SimBA/Mufasa
#   implementation), and, there's no conceptual need to dictate that "normalisation always means subtracting a mean and
#   dividing by a standard deviation". However, to play nicely with some other methods/packages that do view it that
#   way, this returns the mean and the dev along with the actual function applying them.
def generate_normalisation_func(normalisation_algo):
    if normalisation_algo == 'imagenet_common':
        mean = IMAGENET_COMMON_MEAN
        std = IMAGENET_COMMON_STD
    elif normalisation_algo == 'cifar_common':
        mean = CIFAR_COMMON_MEAN
        std = CIFAR_COMMON_STD
    elif normalisation_algo == 'cifar_basic':
        mean = CIFAR_BASIC_MEAN
        std = CIFAR_BASIC_STD
    elif normalisation_algo == 'mnist_common':
        mean = MNIST_BASIC_MEAN
        std = MNIST_BASIC_STD
    elif normalisation_algo == 'none':  # Note that 'none' assumes 3-channel data.
        mean = [0, 0, 0]
        std = [1, 1, 1]
    else:
        raise ValueError("You've specified an invalid normalisation algorithm.")

    def normalisation_func(imgs):
        normalised_imgs = imgs.clone()
        if normalisation_algo == 'mnist_common':
            normalised_imgs = (normalised_imgs - mean[0]) / std[0]
        else:
            if imgs.dim() == 3:
                for i in range(normalised_imgs.size(0)):
                    normalised_imgs[i, :, :] = (normalised_imgs[i, :, :] - mean[i]) / std[i]
            else:
                for i in range(normalised_imgs.size(1)):
                    normalised_imgs[:, i, :, :] = (normalised_imgs[:, i, :, :] - mean[i]) / std[i]
        return normalised_imgs

    return normalisation_func, mean, std


def generate_data_transform(transform_type):
    if transform_type == 'cifar_common_32':
        image_width = 32
        data_transform = transforms.Compose([
            transforms.ToTensor()])
    elif transform_type == 'imagenet_common_224':
        image_width = 224
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_width),
            transforms.ToTensor()])
    elif transform_type == 'imagenet_inception_299':
        image_width = 299
        data_transform = transforms.Compose([
            transforms.Resize(342),
            transforms.CenterCrop(image_width),
            transforms.ToTensor()])
    else:
        raise ValueError("You've specified an invalid data transform.")
    return data_transform, image_width


# This takes in a collection of names and numbers defining the network and data (including sample count) to be read in,
#   and reads it all in, returning the net, images, labels, and all the random stuff (sizes, transforms) that'll be
#   needed.
# sampled_image_dir and dataset_dir come in as pathlib.Path objects.
# It uses caching: if a matching cache file is found, the data will be read directly from there (which both speeds
#   things up when it's available and allows experimental control where appropriate).
# The interdependency between the net and the data samples comes in because here, we are guaranteeing a set that is
#   both sampled without replacement and correctly classified by the network. So the network's behaviour is part of the
#   definition of a valid set in this context. This is why we have one big input function that sets everything up:
#   because the sampler inherently needs access to the net and normalisation algorithm.
def retrieve_net_and_data_samples(sampled_image_dir, prefix, dataset_name, data_transform_name, model_name,
                                  normalisation_algo_name, num_samples, dataset_dir, correct_samples_only,
                                  batch_size=25):
    # This loader first tries to load the provided model by treating it as a path/filename, locally. If that doesn't
    #   work, it will pass it, as is, to torchvision.models. So, the way it works should be pretty obvious, just, don't
    #   have some other model present in the working directory with an identical name to one in torchvision, unless you
    #   intend to load it instead.
    # Note that if using a custom net, it's assumed that you've provided the source file of the network within the
    #   project structure, and that the (.pt) file specified by 'model_name' below contains the name of the module in
    #   the 'architecture_module' field (e.g. 'src.architectures.resnet') and the name of the class in that module in
    #   'architecture_class' (e.g. 'resnet32'), as well as the state_dict in 'state_dict'.
    try:
        print('> Attempting to load {} as a saved model...'.format(model_name))
        loaded_state = torch.load(Path(model_name))
        # Using dynamic importing to load the architecture constructor specific to the saved net:
        architecture_module = importlib.import_module(loaded_state['architecture_module'])
        architecture_constructor = getattr(architecture_module, loaded_state['architecture_class'])
        # Move to GPU:
        model = architecture_constructor().cuda()
        # Hack to work around the fact that, when saved, resnet prepends 'module.' to all the layers:
        if 'resnet' in loaded_state['architecture_module']:
            loaded_state['state_dict'] = {k[7:]: v for (k, v) in loaded_state['state_dict'].items()}

        # Load trained parameters from saved state_dict:
        model.load_state_dict(loaded_state['state_dict'])

    except FileNotFoundError:
        print('> Model NOT FOUND. Attempting to load {} from torchvision model zoo...'.format(model_name))
        model = getattr(models, model_name)(pretrained=True).cuda()

    model.eval()

    normaliser, mean, std = generate_normalisation_func(normalisation_algo_name)
    data_transform, image_width = generate_data_transform(data_transform_name)
    # This is a block copied more or less verbatim from the original repo, which first checks, based on a path, choice
    #   of model, and requested number of runs (i.e. the sample image count), whether a corresponding sample set has
    #   been constructed and saved before, and, if so, just loads that one.
    # The main advantage here is in being able to control experiments by using exactly the same set of images across
    #   runs. Also, you don't have to redo the step of making sure that all sampled images are originally correctly
    #   classified by the model (to do away with the conceptual complications of defining and interpreting
    #   adversariality when that isn't the case).
    # Note: If you do something like redefining your net or normalisation algorithm, *without* renaming them, you can
    #   end up with an invalid cache (i.e. one with images that your new normalisation-net pipeline misclassifies) that
    #   is nonetheless used. I actually had this bug come up. I'm including the normalisation and transform algo names
    #   in the cache filename to mitigate the likelihood of this, but also, don't do stuff like that.
    sampled_image_dir.mkdir(parents=True, exist_ok=True)
    model_name = clean_model_name(model_name)
    if correct_samples_only:
        sample_image_cache = sampled_image_dir / (f"{prefix}{dataset_name}_{data_transform_name}_{model_name}_"
                                                  f"{normalisation_algo_name}_{num_samples}.pth")
    else:
        sample_image_cache = sampled_image_dir / f"{prefix}{dataset_name}_{num_samples}.pth"
    if sample_image_cache.is_file():
        print('> Loading sample set {} from cache...'.format(sample_image_cache))
        checkpoint = torch.load(sample_image_cache)
        images = checkpoint['images']
        labels = checkpoint['labels']
        num_classes = checkpoint['num_classes']
        image_size = checkpoint['image_size']
    else:
        print('> Sample set {} NOT FOUND in cache, loading {} dataset from disk...'.format(sample_image_cache,
                                                                                           dataset_name))
        if dataset_name == 'mnist':
            image_channels = 1
            dataset = datasets.MNIST(dataset_dir, train=False, transform=None, download=True)
        elif dataset_name == 'cifar10_train':
            image_channels = 3
            dataset = datasets.CIFAR10(dataset_dir, train=True, transform=data_transform, download=True)
        elif dataset_name == 'cifar10_test':
            image_channels = 3
            dataset = datasets.CIFAR10(dataset_dir, train=False, transform=data_transform, download=True)
        elif dataset_name == 'imagenet_train':
            # Note that we are using the same data_transform used for the validation set. Normally, random cropping is
            #   used for ImageNet training.
            image_channels = 3
            dataset = datasets.ImageNet(dataset_dir, split='train', transform=data_transform)
        elif dataset_name == 'imagenet_val':
            # Note: Torchvision's (>= 0.7.0) ImageNet loader should allow you to simply place the three tar files
            #   ILSVRC2012_img_val.tar, ILSVRC2012_img_train.tar, and ILSVRC2012_devkit_t12.tar.gz into the data folder,
            #   and do all unpacking and processing for you. The process of doing this with earlier versions is
            #   different, and is strongly advised against.
            image_channels = 3
            dataset = datasets.ImageNet(dataset_dir, split='val', transform=data_transform)
        elif dataset_name == 'image_folder':
            image_channels = 3
            dataset = datasets.ImageFolder(dataset_dir, data_transform)
        else:
            raise ValueError("You've specified an invalid dataset.")
        num_classes = len(dataset.classes)
        image_size = torch.Size((image_channels, image_width, image_width))
        images = torch.zeros(num_samples, image_size[0], image_size[1], image_size[2])
        labels = torch.zeros(num_samples).long()
        preds = labels + 1  # Just initialise all predictions to be wrong.

        if dataset_name == 'imagenet_train':
            sample_indices = list(range(len(dataset)))
            random.shuffle(sample_indices)
        elif dataset_name == 'imagenet_val':
            sample_indices = deque(get_model_specific_indices(model_name))
        else:
            raise NotImplementedError

        while (preds.ne(labels)).any():
            idx = preds.ne(labels).nonzero().flatten()
            for i in list(idx):
                try:
                    images[i], labels[i] = dataset[sample_indices.pop()]
                except IndexError:
                    os.sys.exit("You've requested more samples than there are correctly classified examples in "
                                "your dataset. Reduce your sample count for this dataset.")
            # Note that labels[idx] is a dummy argument here IRL, because the probs return is being ignored.
            # The values of the energy-defining arguments are irrelevant here, because the return is being ignored.
            # They're all being set to False.
            if correct_samples_only:
                preds[idx], _ = get_preds_and_scores(
                    model, images[idx], labels[idx], normaliser, apply_softmax=False, targeted_scores=False,
                    subtract_competition=False, batch_size=batch_size
                )
            else:  # A simple hack that means that the loop will immediately terminate with whichever samples it picked,
                # correctly classified or not:
                preds = labels
        torch.save({'images': images, 'labels': labels, 'num_classes': num_classes, 'image_size': image_size},
                   sample_image_cache)
    return model, images, labels, num_classes, image_size, normaliser, mean, std, sample_image_cache


# This uses Mufasa's canonical outputs, along with the relevant configuration information, to generate the actual
#   perturbation images that can be applied directly to the source images as attacks.
# It's assumed, since you're regenerating images here, that you are doing this in reasonably sized batches on your end.
# This includes the assumption that canonical_adv_mat and attack_basis_ind_mat are regular dense matrices (2D tensors),
#   sliced from the returned sparse structures.
def regenerate_adversarial_perturbations(canonical_adv_mat, attack_basis_ind_mat, attack_to_image_space_transform,
                                         epsilon, norm_bound, attack_mode):
    num_images = canonical_adv_mat.size(0)
    image_size = attack_to_image_space_transform(0).size()  # Just taking the first basis image's size.
    attack_images = torch.zeros([num_images] + list(image_size))
    for col in range(canonical_adv_mat.size(1)):
        this_it_coeffs = canonical_adv_mat[:, col]
        this_it_basis_inds = attack_basis_ind_mat[:, col]
        for coeff in this_it_coeffs.nonzero():
            attack_images[coeff, :, :, :] += (this_it_coeffs[coeff] *
                                              attack_to_image_space_transform(this_it_basis_inds[coeff]))
    # Note that PyTorch has an arguably wacky definition of what a "2-norm" is: it's always just the Euclidean length
    # of the *flattened* version of the tensor, i.e. a "multidimensional Frobenius norm". Fine for our purposes here,
    # but...
    if attack_mode == 'linf_rescale':
        attack_norms = attack_images.abs().amax(dim=(1, 2, 3,))
        attack_images[attack_norms != 0] /= attack_norms[attack_norms != 0, None, None, None]
        attack_images *= epsilon
    elif attack_mode == 'linf_signs':
        attack_images.sign_()
        attack_images *= epsilon
    elif attack_mode == 'standard_simba':
        attack_images *= epsilon
        attack_norms = attack_images.norm(dim=(1, 2, 3), p=2)
        attack_images[attack_norms > norm_bound] /= (
                attack_norms[attack_norms > norm_bound, None, None, None] / norm_bound)
    else:  # <- invalid attack mode option
        raise ValueError("You've specified an invalid attack mode.")
    return attack_images


# Because of the poor state of PyTorch's sparse matrix package as of this writing, it can be necessary to convert over
#   to SciPy in order to get anything done with the result structures. This can help.
# NOTE: This is in principle dangerous, since it breaks the exposed API. In a perfect world, you would never find
#   yourself doing this sort of thing, but, see above.
def convert_sparse_tensor_to_scipy(sparse_torch_mat):
    values = sparse_torch_mat._values()
    indices = sparse_torch_mat._indices()
    return scisp.coo_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=sparse_torch_mat.shape)


# This is intended for drawing/saving basis directions and/or adversarial perturbations as images.
# The input direction is expected to be a PyTorch tensor in CxHxW format.
def visualise_direction(im_space_dir, output_path):
    dims = im_space_dir.size()
    if len(dims) != 3 or dims[0] not in [1, 3]:
        raise ValueError("Invalid input to visualise_directions: must be Torch tensor in [C, H, W] format.")
    im_space_dir_norm = (im_space_dir - im_space_dir.min()) / (im_space_dir.max() - im_space_dir.min())
    plt.imsave(output_path, im_space_dir_norm.permute(1, 2, 0).numpy())

