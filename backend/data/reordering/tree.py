class TreeNode(object):
    """Class that represents k-ary tree node.

        id(int)

        children(list)
    """
    def __init__(self, id=None, name=None):
        self.id = id
        self.parent = None
        self.name = name
        self.leaves = []
        self.children = []
    
    def getLeavesId(self):
        """Return a tuple of sorted id of leaves.
        """
        return tuple(sorted([v.id for v in self.leaves]))

    def isLeaf(self):
        return len(self.children) == 0

    def preOrder(self):
        """Return pre order list of leaves id.
        """
        stack = [self]
        preorder = []
        while len(stack) > 0:
            nd = stack.pop()
            ndid = nd.id
            if nd.isLeaf():
                preorder.append(ndid)
            else:
                for child in reversed(nd.children):
                    stack.append(child)
        return preorder


