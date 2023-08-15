# 6.1 Algorithms

# 1. Write a Python function to recursively read a JSON file.

"****"
# 2. Implement an  O(NlogN) sorting algorithm, preferably quick sort or merge sort.

.sort()

# 3. Find the longest increasing subsequence in a string.

"****"


# 5. Traverse a tree in pre-order, in-order, and post-order.

class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def inorder(root):
    if not root:
        return
    
    inorder(root.left)
    print(root.val)
    inorder(root.right)

def preorder(root):
    if not root:
        return
    
    print(root.val)
    inorder(root.left)
    inorder(root.right)
    

def postorder(root):
    if not root:
        return
    
    inorder(root.left)
    inorder(root.right)
    print(root.val)



# 10. Given a string of mathematical expression, such as 10 * 4 + (4 + 3) / (2 - 1), calculate it. It should support four operators +, -, :, /, and the brackets ().