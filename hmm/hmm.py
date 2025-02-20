import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}
        N_obs = self.observation_states.shape[0]

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        N_hidden = self.hidden_states.shape[0]
        
        self.prior_p= prior_p
        if not (self.prior_p.shape == (N_hidden, ) and 
                np.all(self.prior_p >= 0) and
                np.abs(np.sum(self.prior_p) - 1) < 0.001):
            raise ValueError("Invalid prior distribution")

        self.transition_p = transition_p
        if not (self.transition_p.shape == (N_hidden, N_hidden) and 
                np.all(self.transition_p >= 0) and 
                np.all(np.abs(np.sum(self.transition_p, axis=1) - 1) < 0.001)):
            raise ValueError("Invalid transition distribution")
        
        self.emission_p = emission_p
        if not (self.emission_p.shape == (N_hidden, N_obs) and 
                np.all(self.emission_p >= 0) and 
                np.all(np.abs(np.sum(self.emission_p, axis=1) - 1) < 0.001)):
            raise ValueError("Invalid emission distribution")


    def forward(self, input_observation_states: np.ndarray) -> float:
        """

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """

        if len(input_observation_states.shape) != 1:
            raise ValueError("Invalid observation states shape")
        if any(obs not in self.observation_states_dict for obs in input_observation_states):
            raise ValueError("Invalid input observation states")
        input_observation_indices = [self.observation_states_dict[obs] 
                                     for obs in input_observation_states]
        
        # Step 1. Initialize variables
        N = len(self.hidden_states)
        T = len(input_observation_states)
        # alpha[j, t] = p(o[1 to t], hidden at t = j)
        alpha = np.zeros((N, T))
        # Initialize time 1
        for hidden in range(N):
            alpha[hidden, 0] = self.prior_p[hidden] * self.emission_p[hidden, input_observation_indices[0]]

        # Step 2. Calculate probabilities
        for time in range(1, T):
            for hidden in range(N):
                for hidden_prev in range(N):
                    alpha[hidden, time] += \
                        (alpha[hidden_prev, time-1] * 
                         self.transition_p[hidden_prev, hidden] * 
                         self.emission_p[hidden, input_observation_indices[time]])

        # Step 3. Return final probability 
        return np.sum(alpha[:, T-1])


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """
        if len(decode_observation_states.shape) != 1:
            raise ValueError("Invalid observation states shape")
        if any(obs not in self.observation_states_dict for obs in decode_observation_states):
            raise ValueError("Invalid input observation states")
        decode_observation_indices = [self.observation_states_dict[obs] 
                                     for obs in decode_observation_states]
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        N = len(self.hidden_states)
        T = len(decode_observation_states)
        # v[j, t] = max_{hidden[1 to t-1]} p(hidden[1 to t-1], hidden t = j, o[1 to t])
        v = np.zeros((N, T))
        for hidden in range(N):
            v[hidden, 0] = self.prior_p[hidden] * self.emission_p[hidden, decode_observation_indices[0]]
        # b[j, t] = best previous state to generate v[j, t]
        b = np.zeros((N, T-1))   
        # No initialization for b, as time 0 has no prev     
       
       # Step 2. Calculate Probabilities
        for time in range(T):
            for hidden in range(N):
                for hidden_prev in range(N):
                    prob = (v[hidden_prev, time-1] * 
                            self.transition_p[hidden_prev, hidden] * 
                            self.emission_p[hidden, decode_observation_indices[time]])
                    if prob > v[hidden, time]:
                        v[hidden, time] = prob
                        b[hidden, time-1] = hidden_prev
            
        # Step 3. Traceback
        # The final hidden state is the one with the largest probability of ending there
        reversed_traceback = [int(np.argmax(v[:, T-1]))]
        # Follow the backpointers back to time 0
        for time in range(T-2, -1, -1):
            reversed_traceback.append(int(b[reversed_traceback[-1], time]))

        # Step 4. Return best hidden state sequence
        return [self.hidden_states_dict[index] for index in reversed_traceback[::-1]]
        