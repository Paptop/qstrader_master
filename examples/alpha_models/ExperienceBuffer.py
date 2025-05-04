from collections import deque
import random

class ExperienceBuffer:
    def __init__(self, capacity):
        """
        Initialize the experience buffer.

        Parameters
        ----------
        capacity : int
            The maximum number of experiences to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, current_state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Parameters
        ----------
        current_state : np.array
            The current state of the environment.
        action : int
            The action taken in the current state.
        reward : float
            The reward received after taking the action.
        next_state : np.array
            The next state of the environment.
        done : bool
            Whether the episode has ended.
        """
        self.buffer.append((current_state, action, reward, next_state, done))

    def sample(self, batch_size, sequence_length=12):
        """
        Sample a batch of temporally contiguous sequences of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of sequences to sample.
        sequence_length : int
            The length of each sequence.

        Returns
        -------
        list
            A list of sampled sequences, where each sequence is a list of experiences.
        """
        sequences = []
        buffer_size = len(self.buffer)

        if buffer_size < sequence_length:
            raise ValueError("Not enough experiences in the buffer to sample a sequence.")

        for _ in range(batch_size):
            # Randomly select a starting index such that the sequence fits in the buffer
            start_index = random.randint(0, buffer_size - sequence_length)
            # Extract a contiguous sequence of experiences
            sequence = list(self.buffer)[start_index:start_index + sequence_length]
            sequences.append(sequence)

        return sequences

    def size(self):
        """
        Get the current size of the buffer.

        Returns
        -------
        int
            The number of experiences in the buffer.
        """
        return len(self.buffer)