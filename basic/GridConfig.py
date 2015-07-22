class GridConfig:
    def __init__(self, priority, value):
        self.priority = priority
        self.value = value

    def __cmp__(self, other):
        return cmp(self.priority, other.priority)

    def cmp(sum_max_x, sum_max_y):
        if sum_max_x[0] > sum_max_y[0]:
            return 1
        elif sum_max_x[0] == sum_max_y[0]:    
            if sum_max_x[1] > sum_max_y[1]:
                return 1

        return -1
                
