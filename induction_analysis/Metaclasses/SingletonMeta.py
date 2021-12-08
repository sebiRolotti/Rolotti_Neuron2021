from threading import Lock


class SingletonMeta(type):
    """
    Thread Safe implementation of the Singleton Creation Design Pattern.
    """
    _instance = None  # instance to be hold
    _lock = Lock()  # the lock

    # initializer with call
    def __call__(cls, *args, **kwargs):
        with cls._lock:  # make sure that only an instance is created regardless of the Threads
            if cls._instance is None:  # check and create a new instance, if it hasn't been initialized
                cls._instance = super(SingletonMeta, cls).__call__(*args, **kwargs)

        # return the existing instance otherwise.
        return cls._instance
