import collections

class CustomDict(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key

class CombiGridDict(CustomDict):
    def get_priority_level(self):
        if self.store:
            priority_key = tuple([0])
            for key, value in self.store.iteritems():
                if value.grid_refinement_list:
                    if sum(priority_key) < sum(key):
                        priority_key = key
                    elif sum(priority_key) == sum(key):
                        if max(priority_key) > max(key):
                            priority_key = key

            return priority_key

        return None

    def get_refine_count(self):
        count = 0
        if self.store:
            for key, value in self.store.iteritems():
                if value.grid_refinement_list:
                    count += 1

        return count 
