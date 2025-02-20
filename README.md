# HW6-HMM

![BuildStatus](https://github.com/isaiahah/HW6-HMM/actions/workflows/test.yml/badge.svg)

## Forward Algorithm
The forward algorithm calculates the probability of a series of `T` observed states, agnostic of the `N` possible hidden states. It performs this using dynamic programming:
* It creates an array `alpha` with `N` rows and `T` columns where `alpha[n, t]` represents the probability for the first `t` observed states given that the hidden state at time `t` is `n`.
* For `t = 0`, this denotes the first hidden state is `n` and it emits the first observation. Thus, `alpha[n, t] = prior(n) * emission(n, obs[0])`.
* For all larger `t`, the hidden state at that time transitioned from any prior state to `n` and emitted the `t`th observation. Thus, `alpha[n, t] = sum_{prev_n} alpha[prev_n, t-1] * transition(prev_n, n) * emission(n, obs[t])`.
* The final probability of all observation states is the sum of all `alpha[n, T]`, as we consider the probabilities of ending in any final hidden state.


## Viterbi Algorithm
The viterbi algorithm calculates the most likely series of hidden states to generate a series of observations. It performs this using dynamic programming:
* It creates an array `v` with `N` rows and `T` columns where `v[n, t]` represents the probability of the most likely series of hidden states ending at state `n` in time `t` which produce the first `t` observed states.
* For `t = 0`, this denotes the first hidden state is `n` and it emits the first observation. Thus, `alpha[n, t] = prior(n) * emission(n, obs[0])`.
* For all larger `t`, the hidden state at that time transitioned from some prior state to `n` and emitted the `t`th observation. Thus, `alpha[n, t] = max_{prev_n} v[prev_n, t-1] * transition(prev_n, n) * emission(n, obs[t])`.
* In all updates, it uses the matrix `b` to store the previous state that the most likely path took.
* The final most likely series of hidden states is derived by starting from the hidden state with largest `alpha[n, T]` and following the backtrace backwards.


## Test Cases
The tests verify the forward algorithm returns a reasonable value for the probability of the input sequence and the Viterbi algorithm returns the expected most likely series of states.

The code has checks to verify several assumptions on the data quality:
* The probability matrices should have dimensions matching the number of hidden and observed states.
* The probability matrices should have positive values only.
* The probability matrices should sum to 1 along the relevant axes which indicate the possible outcomes for a single condition.
* The input series of observed states should be a single-dimension array with only possible observed state values.

If any of these conditions is false, it raises a `ValueError`. The tests verify this error is raised when appropriate.
