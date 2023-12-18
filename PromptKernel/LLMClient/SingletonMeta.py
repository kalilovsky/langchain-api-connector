class SingletonMeta(object):
    _instance = dict()

    def __new__(cls, model):
        if cls._instance.get(model) is None:
            cls._instance[model] = object.__new__(cls)
        return cls._instance[model]
