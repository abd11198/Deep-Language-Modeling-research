class Node:
    """
    Node of the linked list.
    """
    def __init__(self, data):
        self.data = data  # The data (MarkovNode in this case)
        self.next = None  # Pointer to the next node


class LinkedList:
    """
    A simple Linked List class to hold MarkovNodes.
    """
    def __init__(self):
        self.first = None  # First node in the linked list
        self.last = None   # Last node in the linked list
        self.size = 0      # Size of the linked list

    def add(self, data):
        """
        Add data to a new node at the end of the linked list.

        :param data: Pointer to dynamically allocated data
        :return: 0 on success, raises an exception on failure
        """
        new_node = Node(data)  # Create a new node

        if not new_node:
            raise MemoryError("Failed to allocate memory for a new Node.")

        if self.first is None:
            # If the list is empty, set both first and last to the new node
            self.first = new_node
            self.last = new_node
        else:
            # Append the new node at the end and update last
            self.last.next = new_node
            self.last = new_node

        self.size += 1
        return 0
