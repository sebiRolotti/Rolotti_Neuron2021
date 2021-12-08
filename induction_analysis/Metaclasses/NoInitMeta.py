class NoInitMeta(type):
    def __new__(cls, *args, **kwargs):
        return cls.__new__(cls)
