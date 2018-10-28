# This function allows to estimate the initial probability, and the emission and
# transition matrices through the observable sequence and the hidden sequence
# Arguments:
#       observable.sequence: the observable sequence
#       hidden.sequence: the hidden sequence
#       [hidden.alphabet]: the alphabet of the hidden sequence, by default c("1","2")
#       [observed.alphabet]: the alphabet of the observed sequence,
#                            by default c(""a","c","g","t")
# Returns:
#       A list of three elements: the initial probability, the emission matrix
#       and the transition matrix
getEstimatedMatrices <- function(observable.sequence,hidden.sequence,
                                 hidden.alphabet=c("1","2"),
                                 observed.alphabet=c("a","c","g","t"))
{
  hidden.states.number <- length(hidden.alphabet)
  observed.states.number <- length(observed.alphabet)

  # calculates the initial probability of the hidden sequence through the frequency
  # of each hidden state
  initial.probabilities.estimate <- table(hidden.sequence)/length(hidden.sequence)

  # generates the transition matrix for any number of states
  pair.occurrences = count(hidden.sequence,2,alphabet=hidden.alphabet)
  transition.estimate <- matrix(pair.occurrences,nrow=hidden.states.number,
                                ncol=hidden.states.number)
  for(i in 1:hidden.states.number){
        transition.estimate[i,] <- transition.estimate[i,]/sum(transition.estimate[i,])
  }

  # generates the emission matrix for any number of hidden states and four observable states
  times = hidden.states.number * observed.states.number
  emission.estimate <- matrix(rep(0,times),nrow=hidden.states.number,
                              ncol=observed.states.number)

  # for each hidden state is extracted the corresponding observable subsequence
  subsequence.state <- vector("list", hidden.states.number)
  for(i in 1:hidden.states.number){
        subsequence.state[[i]] <- observable.sequence[hidden.sequence == i]
  }
  # fills each row of the emission matrix with the frequencies of its respective subsequence
  for(i in 1:hidden.states.number){
        if(length(subsequence.state[[i]]) > 0)
        {
          emission.estimate[i,] <- getFrequency(subsequence.state[[i]],
                                        observed.alphabet)/length(subsequence.state[[i]])
        }
  }

  # sets up the result set
  result <- list(emission=emission.estimate, transition=transition.estimate,
                 initial=initial.probabilities.estimate)

  return(result)
}
