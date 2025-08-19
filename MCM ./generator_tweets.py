import random
import sys

MAX_LINE_LENGTH = 10000
MAX_WORD_LENGTH = 100
DEFAULT_FILE = "justdoit_tweets.txt"


class Node:
    """
    Node for the linked list in the Markov Chain database.
    """
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """
    Simple linked list implementation for the Markov Chain database.
    """
    def __init__(self):
        self.first = None
        self.last = None
        self.size = 0

    def add(self, data):
        """
        Add a new node to the linked list.
        :param data: The data to store in the node.
        :return: The new node.
        """
        new_node = Node(data)
        if not self.first:
            self.first = new_node
        else:
            self.last.next = new_node
        self.last = new_node
        self.size += 1
        return new_node


class MarkovNodeFrequency:
    """
    Represents the frequency of a transition from one MarkovNode to another.
    """
    def __init__(self, next_word, frequency=1):
        self.next_word = next_word
        self.frequency = frequency


class MarkovNode:
    """
    Node in the Markov Chain.
    """
    def __init__(self, data):
        self.data = data
        self.list_frequencies = []
        self.frequencies_size = 0

    def add_frequency(self, next_node):
        """
        Add or update a frequency for a transition to the next node.
        :param next_node: The MarkovNode to add a transition to.
        :return: None
        """
        for freq_entry in self.list_frequencies:
            if freq_entry.next_word == next_node:
                freq_entry.frequency += 1
                return
        self.list_frequencies.append(MarkovNodeFrequency(next_node))
        self.frequencies_size += 1


class MarkovChain:
    """
    Represents the Markov Chain, containing a database of nodes.
    """
    def __init__(self):
        self.database = LinkedList()

    def add_to_database(self, data):
        """
        Add a new word to the database or return the existing node.
        :param data: The word to add.
        :return: The Node containing the word.
        """
        current = self.database.first
        while current:
            if current.data.data == data:
                return current
            current = current.next

        new_node = MarkovNode(data)
        return self.database.add(new_node)

    def add_node_to_frequencies_list(self, first_node, second_node):
        """
        Add or update the frequency of a transition from one node to another.
        :param first_node: The source MarkovNode.
        :param second_node: The target MarkovNode.
        :return: None
        """
        first_node.add_frequency(second_node)

    def get_random_node(self):
        """
        Get a random starting node from the database.
        :return: A random MarkovNode.
        """
        if self.database.size == 0:
            return None
        random_index = random.randint(0, self.database.size - 1)
        current = self.database.first
        for _ in range(random_index):
            current = current.next
        return current.data

    def generate_tweet(self, start_node=None, max_words=100):
        """
        Generate a tweet starting from a given node.
        :param start_node: The starting node.
        :param max_words: Maximum number of words in the tweet.
        :return: None
        """
        if not start_node:
            start_node = self.get_random_node()
        if not start_node:
            return

        tweet = [start_node.data]
        current_node = start_node
        for _ in range(max_words - 1):
            next_node = self.get_next_random_node(current_node)
            if not next_node:
                break
            tweet.append(next_node.data)
            if next_node.data.endswith('.'):
                break
            current_node = next_node

        print(" ".join(tweet))

    def get_next_random_node(self, current_node):
        """
        Select the next node based on transition frequencies.
        :param current_node: The current MarkovNode.
        :return: The next MarkovNode.
        """
        if not current_node.list_frequencies:
            return None

        total_frequency = sum(freq.frequency for freq in current_node.list_frequencies)
        random_value = random.randint(0, total_frequency - 1)
        cumulative = 0
        for freq_entry in current_node.list_frequencies:
            cumulative += freq_entry.frequency
            if random_value < cumulative:
                return freq_entry.next_word
        return None


def fill_database(file, words_to_read, markov_chain):
    words_read = 0
    previous_node = None
    flag_frequency = True

    for line in file:
        temp_words = line.split()
        for temp_word in temp_words:
            if len(temp_word) > MAX_WORD_LENGTH:
                continue

            current_node = markov_chain.add_to_database(temp_word)
            if previous_node and flag_frequency:
                markov_chain.add_node_to_frequencies_list(previous_node.data, current_node.data)

            if temp_word.endswith('.'):
                flag_frequency = False
            else:
                flag_frequency = True

            previous_node = current_node
            words_read += 1

            if 0 < words_to_read <= words_read:
                return
        previous_node = None


def main():
    seed = 1
    tweets_to_generate = 50
    file_path = "justdoit_tweets.txt"
    words_to_read = 10000

    random.seed(seed)

    try:
        with open(file_path, "r") as file:
            markov_chain = MarkovChain()
            fill_database(file, words_to_read, markov_chain)
            for i in range(tweets_to_generate):
                print(f"Tweet {i + 1}: ", end="")
                markov_chain.generate_tweet(None, MAX_WORD_LENGTH)
    except FileNotFoundError:
        print(f"Error: Cannot open file '{file_path}'", file=sys.stderr)
        exit(1)
    except MemoryError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        exit(1)


if __name__ == "__main__":
    main()
