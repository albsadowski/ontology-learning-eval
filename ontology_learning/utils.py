from functools import wraps


def debuglog(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        debug = kwargs.pop("debug", False)
        if debug:
            print(f"E {func.__name__}")
        res = func(*args, **kwargs)
        if debug:
            print(f"X {func.__name__}")
            print(f"Result: {res}")
        return res

    return wrapper
