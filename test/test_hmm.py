import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """
    # Load parameters from data files
    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    # Initialize the model
    mini_hmm_model = HiddenMarkovModel(mini_hmm["observation_states"],
                                       mini_hmm["hidden_states"],
                                       mini_hmm["prior_p"],
                                       mini_hmm["transition_p"],
                                       mini_hmm["emission_p"])

    # Load the observation states and the best sequence of hidden states
    observation_states = mini_input["observation_state_sequence"]
    best_hidden_states = mini_input["best_hidden_state_sequence"]

    forward_prob = mini_hmm_model.forward(observation_states)
    assert np.abs(forward_prob - 0.03506441162109375) < 0.001

    # Verify the Viterbi algorithm
    viterbi_hidden = mini_hmm_model.viterbi(observation_states)
    assert len(viterbi_hidden) == observation_states.shape[0]
    for time in range(observation_states.shape[0]):
        assert viterbi_hidden[time] == best_hidden_states[time]

    # Check edge case: invalid probability matrices shape
    with pytest.raises(ValueError):
        bad_observation_states = mini_hmm["observation_states"][:-1]
        mini_hmm_model = \
            HiddenMarkovModel(bad_observation_states,
                              mini_hmm["hidden_states"],
                              mini_hmm["prior_p"],
                              mini_hmm["transition_p"],
                              mini_hmm["emission_p"])
    with pytest.raises(ValueError):
        bad_hidden_states = mini_hmm["hidden_states"][:-1]
        mini_hmm_model = \
            HiddenMarkovModel(mini_hmm["observation_states"],
                              bad_hidden_states,
                              mini_hmm["prior_p"],
                              mini_hmm["transition_p"],
                              mini_hmm["emission_p"])
    # Check edge case: probability matrices do not sum to 1
    with pytest.raises(ValueError):
        bad_prior_p = np.copy(mini_hmm["prior_p"])
        bad_prior_p[-1] = 0
        mini_hmm_model = \
            HiddenMarkovModel(mini_hmm["observation_states"],
                              mini_hmm["hidden_states"],
                              bad_prior_p,
                              mini_hmm["transition_p"],
                              mini_hmm["emission_p"])
    with pytest.raises(ValueError):
        bad_transition_p = np.copy(mini_hmm["transition_p"])
        bad_transition_p[-1, -1] = 0
        mini_hmm_model = \
            HiddenMarkovModel(mini_hmm["observation_states"],
                              mini_hmm["hidden_states"],
                              mini_hmm["prior_p"],
                              bad_transition_p,
                              mini_hmm["emission_p"])
    with pytest.raises(ValueError):
        bad_emission_p = np.copy(mini_hmm["emission_p"][:-1])
        bad_emission_p[-1, -1] = 0
        mini_hmm_model = \
            HiddenMarkovModel(mini_hmm["observation_states"],
                              mini_hmm["hidden_states"],
                              mini_hmm["prior_p"],
                              mini_hmm["transition_p"],
                              bad_emission_p)

    # Check edge case: invalid observation states
    with pytest.raises(ValueError):
        bad_observation_states = np.array([observation_states, observation_states])
        forward_prob = mini_hmm_model.forward(bad_observation_states)
    with pytest.raises(ValueError):
        bad_observation_states = np.copy(observation_states)
        bad_observation_states[-1] = "bad"
        forward_prob = mini_hmm_model.forward(bad_observation_states)

def test_full_weather():

    """
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    # Load parameters from data files
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    # Initialize the model
    full_hmm_model = HiddenMarkovModel(full_hmm["observation_states"],
                                       full_hmm["hidden_states"],
                                       full_hmm["prior_p"],
                                       full_hmm["transition_p"],
                                       full_hmm["emission_p"])

    # Load the observation states and the best sequence of hidden states
    observation_states = full_input["observation_state_sequence"]
    best_hidden_states = full_input["best_hidden_state_sequence"]

    forward_prob = full_hmm_model.forward(observation_states)
    assert np.abs(forward_prob - 1.6864513843961343e-11) < 0.001

    # Verify the Viterbi algorithm
    viterbi_hidden = full_hmm_model.viterbi(observation_states)
    assert len(viterbi_hidden) == observation_states.shape[0]
    for time in range(observation_states.shape[0]):
        assert viterbi_hidden[time] == best_hidden_states[time]
