from Queue import PriorityQueue

class GridPriorityQueue(PriorityQueue):
    def __contains__(self, item):
        for e_item in self.queue:
            if e_item.value[0] == item:
                return True

        return False
