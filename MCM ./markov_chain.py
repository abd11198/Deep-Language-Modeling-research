import random


class MarkovNodeFrequency:
    """
    Represents the frequency of a MarkovNode in the context of another MarkovNode.
    """
    def __init__(self, next_word, frequency=1):
        self.next_word = next_word  # Pointer to the next word (MarkovNode)
        self.frequency = frequency  # Frequency of occurrence


class MarkovNode:
    """
    Represents a node in the Markov Chain.
    """
    def __init__(self, data):
        self.data = data  # The word or data of this node
        self.list_frequencies = []  # List of MarkovNodeFrequency
        self.frequencies_size = 0  # Number of elements in list_frequencies


class LinkedList:
    """
    A simple Linked List class to hold MarkovNodes.
    """
    def __init__(self):
        self.first = None
        self.last = None
        self.size = 0

    def add(self, data):
        new_node = Node(data)
        if not self.first:
            self.first = new_node
        else:
            self.last.next = new_node
        self.last = new_node
        self.size += 1


class Node:
    """
    Node of the linked list.
    """
    def __init__(self, data):
        self.data = data
        self.next = None


class MarkovChain:
    """
    Contains the database of nodes for the Markov Chain.
    """
    def __init__(self):
        self.database = LinkedList()


def get_random_number(max_number):
    return random.randint(0, max_number - 1)


def get_first_random_node(markov_chain):
    if not markov_chain or not markov_chain.database or markov_chain.database.size == 0:
        return None

    random_index = get_random_number(markov_chain.database.size)
    current_node = markov_chain.database.first
    for _ in range(random_index):
        current_node = current_node.next

    return current_node.data


def add_to_database(markov_chain, data):
    if not markov_chain or not data:
        return None

    existing_node = get_node_from_database(markov_chain, data)
    if existing_node:
        return existing_node

    new_markov_node = MarkovNode(data)
    markov_chain.database.add(new_markov_node)
    return markov_chain.database.last


def get_node_from_database(markov_chain, data):
    if not markov_chain or not data or not markov_chain.database:
        return None

    current_node = markov_chain.database.first
    while current_node:
        markov_node = current_node.data
        if markov_node.data == data:
            return current_node
        current_node = current_node.next

    return None


def add_node_to_frequencies_list(first_node, second_node):
    if not first_node or not second_node:
        return False

    # Check if second_node is already in first_node's list_frequencies
    for freq_entry in first_node.list_frequencies:
        if freq_entry.next_word.data == second_node.data:
            freq_entry.frequency += 1
            return True

    # Not found, need to add new MarkovNodeFrequency
    first_node.list_frequencies.append(MarkovNodeFrequency(second_node))
    first_node.frequencies_size += 1
    return True


def free_database(markov_chain):
    if not markov_chain:
        return

    current_node = markov_chain.database.first
    while current_node:
        markov_node = current_node.data
        # Free data and frequencies
        markov_node.list_frequencies.clear()
        current_node = current_node.next

    markov_chain.database = None


def get_next_random_node(state_struct):
    if not state_struct or state_struct.frequencies_size == 0:
        return None

    # Calculate total frequency sum
    total_frequency = sum(freq.frequency for freq in state_struct.list_frequencies)

    # Generate a random number within the total frequency
    random_value = get_random_number(total_frequency)

    # Select the next node based on the random value
    cumulative_frequency = 0
    for freq_entry in state_struct.list_frequencies:
        cumulative_frequency += freq_entry.frequency
        if random_value < cumulative_frequency:
            return freq_entry.next_word

    return None  # Fallback (should not reach here)


def generate_tweet(markov_chain, first_node=None, max_length=10):
    if not markov_chain or max_length <= 0:
        return

    current_node = first_node
    if not current_node:
        current_node = get_first_random_node(markov_chain)
        if not current_node:
            return

    print(current_node.data, end="")

    for _ in range(1, max_length):
        current_node = get_next_random_node(current_node)
        if not current_node:
            break
        print(f" {current_node.data}", end="")

    print()

