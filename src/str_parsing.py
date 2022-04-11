from os.path import split, splitext


def clean_model_name(s):
    '''
    Output clean string from one potentially containing nested structure and extension
    '''

    # if has extension, remove it
    s = splitext(s)[0]
    # pick the tail of the path
    s = split(s)[-1]
    return s


