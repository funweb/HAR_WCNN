import os

def _norm_path(path):
    """
    Decorator function intended for using it to normalize a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """
    print("aaa")
    def normalize_path(*args, **kwargs):
        return os.path.normpath(path(*args, **kwargs))
    return normalize_path


@_norm_path
def kk(path):
    print(path)

if __name__ == '__main__':
    p = os.getcwd()
    print(kk(p))
