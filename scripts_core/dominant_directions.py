# This implements the core approach of "With Friends Like These, Who Needs Adversaries?"
#   (paper: https://arxiv.org/abs/1807.04200, code: https://github.com/torrvision/whoneedsadversaries), that being the
#   composition of a matrix whose columns each represent a different sample of an image-space direction of interest
#   (typically a gradient or the slightly more involved output of a first-order adversarial attack method), and the
#   decomposition of that matrix into its SVD. The (left) singular vectors and values then represent an ordered list of
#   directions of net sensitivity.

# This is very closely related to standard PCA, the difference here basically being that there is no mean subtraction.
#   The reason for this is that the vectors supplied are indeed vectors, not sample points. That is, they represent
#   deviations from an origin which is conceptually/semantically equivalent to the sample mean in the case of PCA.
#   In the case of adversaries/net sensitivity directions, that simply corresponds to "no (additional) signal".
#   Just to be clear, this method is not some novel insight in and of itself: it's used in e.g. multiple papers by
#   Moosavi-Dezfooli and Fawzi. Most recently, a variant of it forms the core of Neural Anisotropy Directions: they
#   use the eigendecomposition of the sum of the vector outer products, which is mathematically identical. There are
#   numerical and computational reasons to favour some representations and decompositions over others, so the
#   implementation remains open in this regard: the key is in the equivalence to the above.

# For many reasons, including/especially space, this method will assume that the adversarial directions have been dumped
#   to a folder, each in its own file (corresponding to the perturbation of a single image). This method's job is to
#   read from the supplied folder, vectorise and stack the inputs, perform the decomposition as above, and write the
#   result back to disc. From there, it can be used and reused by methods that can leverage the decomposition, most
#   particularly, in this case, Mufasa.
# The directions are assumed to be pickled Torch tensors. It's assumed that they may need to be vectorised/reshaped
#   (as they may and probably should be stored to reflect the layout of the image space from which they were derived).

# Note that, as of this writing, there is no size checking of inputs to make sure nothing will explode, and this is a
#   particularly easy one to make explode. As is, you will simply have to ensure that you don't hand this thing more
#   than your machine is capable of handling.

import configargparse
from pathlib import Path

import torch


def dominant_directions():
    parser = configargparse.ArgumentParser(
        description="Method for reading a collection of stored direction vectors and returning the reduced SVD of the "
                    "corresponding column matrix. The primary intended purpose of this is the extraction of dominant "
                    "net sensitivity directions for black-box attacks, in the manner suggested in 'With Friends Like "
                    "These, Who Needs Adversaries?'.",
        add_help=False
    )
    required_named_arguments = parser.add_argument_group("required named arguments")
    required_named_arguments.add_argument(
        '--data_folder', type=str, required=True,
        help="The path to the folder containing the direction vector data. As it stands, the folder is expected to "
             "contain one file per direction: all files in the folder will be assumed to be valid directions, of the "
             "same dimension (behaviour undefined otherwise)."
    )
    optional_arguments = parser.add_argument_group("optional arguments")
    optional_arguments.add_argument(
        '-h', '--help', action='help', default=configargparse.SUPPRESS,
        help="show this help message and exit"
    )
    optional_arguments.add_argument(
        '--resampling_factor', type=float, default=1.0,
        help="In order to manage memory and runtime costs, the user can optionally specify the factor by which the "
             "direction (perturbation) vectors will be resampled before being assembled and decomposed. They will be "
             "resampled to the original spatial resolution before being returned, unless --output_size is explicitly"
             "specified. (default = '%(default)f')"
    )
    optional_arguments.add_argument(
        '--output_size', type=int, default=None,
        help="The desired dimension of each side of the (square) output, for the basis vectors when interpreted as "
             "images. If this option is not specified, the outputs will automatically be resampled back to the "
             "original input dimensions. (default = '%(default)s')"
    )
    optional_arguments.add_argument(
        '--config_file', is_config_file=True,
        help="Optional file from which to read parameter values. In the case of multiple specifications, the override "
             "order is (command line) > (environment vars) > (config file) > (defaults), as in the ConfigArgParse "
             "docs. See the docs for the valid config file format options."
    )
    optional_arguments.add_argument(
        '--data_ext', type=str, default='.pt',
        help="The extension that will be used to identify the files in the data folder assumed to contain input "
             "vectors. The output file shares this extension. (default = '%(default)s')"
    )
    optional_arguments.add_argument(
        '--output_folder', type=str, default='./dominant_directions',
        help="The path to the file in which the SVD results will be stored. The variable names in the saved file will "
             "be 'u' (a matrix) and 's' (a vector). You can provide a local or full path. The filename itself is "
             "constructed from the name of the input data folder, as well as other relevant naming options. "
             "(default = '%(default)s')")

    optional_arguments.add_argument(
        '--output_suffix', type=str, default='',
        help="An optional suffix that can be appended to the output filename (before the extension). "
             "(default = '%(default)s')"
    )

    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    output_folder = Path(args.output_folder)

    if not data_folder.is_dir():
        raise FileNotFoundError("The data folder you've specified doesn't seem to exist.")
    output_folder.mkdir(parents=True, exist_ok=True)

    # For the assumed format (each perturbation in its own file), we determine the dimensions of the assembled matrix
    #   by counting the dimension of a given sample (row dimension) and counting the number of samples (column
    #   dimension). As of this writing, the assumption that all other vectors will match that dimension (i.e. be
    #   compatible with each other) must hold or it's GIGO.
    sample_filenames = list(data_folder.glob(f"*{args.data_ext}"))
    num_samples = len(sample_filenames)

    first_sample = torch.load(sample_filenames[0]).unsqueeze(0)
    # The assumption is that the samples are pickled Torch tensors. They are assumed to be laid out analogously to the
    #   images to which they would be added (C,W,H).
    original_sample_size = first_sample.size()
    # Note some of the details in the interpolation call. These seem like reasonable choices, but they are choices.
    resized_first_sample = torch.nn.functional.interpolate(
        first_sample, scale_factor=args.resampling_factor, mode='bilinear', recompute_scale_factor=True,
        align_corners=False)
    resized_sample_size = resized_first_sample.size()
    sample_dim = resized_first_sample.numel()

    # Preallocation of what may be a very large matrix, depending on your image-space dimension and sample count (the
    #   latter of which will generally need to scale with the former).
    # I'm working with the transpose of the matrix-of-columns that I normally talk about, for memory layout reasons (way
    #   faster to load this). You could just take the SVD of this and return right singular vectors instead, but so as
    #   not to confuse everything I've said before, I'll feed the transpose of this to the SVD.
    master_matrix_t = torch.zeros(num_samples, sample_dim)

    for row_ind, filename in enumerate(sample_filenames):
        im_space_direction = torch.load(filename).unsqueeze(0)

        resized_im_space_dir = torch.nn.functional.interpolate(
            im_space_direction, scale_factor=args.resampling_factor, mode='bilinear', recompute_scale_factor=True,
            align_corners=False)
        direction_vec = resized_im_space_dir.flatten()
        master_matrix_t[row_ind, :] = direction_vec

    # Torch's SVD should be reduced by default:
    u, s, _ = torch.linalg.svd(master_matrix_t.transpose(0, 1), full_matrices=False)

    # The columns of u must all be reinterpreted as images, and resized back to the original resolution, before being
    #   returned:
    # (Note that u comes out of the SVD with images vectorised as columns. For the view/reshaping to work properly, we
    #   must transpose to make the images rows, then transpose again at the end to restore the original convention.)
    # Note also that I have been a little bit wacky in forming a data matrix by stacking columns instead of rows
    #   (I actually did the rows in the end, but then ha ha, transposed), which is why you'll find a lot of PCA/SVD
    #   reference material out there that will refer to a V that plays the role of our U, above. But note also that
    #   in either case, the directions of interest are columns, and because of our special interpretation of those as
    #   quasi-images, you're not really going to be able to get around the transposition mess below, so.
    # For the sake of clarity, I'm going to take these operations more or less step by step:
    u.transpose_(0, 1)
    u = u.view([-1] + list(resized_sample_size[1:]))
    if args.output_size is not None:
        output_size = torch.Size([args.output_size,args.output_size])
    else:
        output_size = original_sample_size[2:]
    u = torch.nn.functional.interpolate(u, size=output_size, mode='bilinear', align_corners=False)
    u = u.reshape([u.size(0), -1])
    u.transpose_(0, 1)

    # NOTE: The "u" that results from the upsampling procedure is no longer an orthonormal matrix. Its columns do not
    #   (outside of special cases) have unit length, and they may not be perfectly orthogonal due to interpolation
    #   artifacts.
    u /= u.norm(dim=[0]).expand_as(u)  # <- normalise columns
    u, _ = torch.linalg.qr(u, mode='reduced')  # <- ensure perfect orthogonality

    # This assumes the folder the examples were stored in described what they were, and reuses that folder name for the
    #   output filename:
    torch.save({'u': u, 's': s}, output_folder / (str(data_folder.name) + args.output_suffix + args.data_ext))


if __name__ == '__main__':
    dominant_directions()
