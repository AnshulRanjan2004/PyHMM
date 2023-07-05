from myhmm_scaled import MyHmmScaled
model_file_name = r"./models/coins.json"
hmm = MyHmmScaled(model_file_name)
print(hmm)
observations = [("Heads", "Tails", "Heads", "Heads", "Heads", "Tails"),
                ("Tails", "Tails", "Tails", "Heads", "Heads", "Tails")]

log_prob_1 = hmm.forward_scaled(observations[0])
log_prob_2 = hmm.forward_scaled(observations[1])

print("log_prob_1", log_prob_1)
print("log_prob_2", log_prob_2)

log_prob_1 = hmm.backward_scaled(observations[0])
log_prob_2 = hmm.backward_scaled(observations[1])

print("log_prob_1", log_prob_1)
print("log_prob_2", log_prob_2)

print("Model bef trg = ", hmm.pi, hmm.A, hmm.B)
hmm.forward_backward_multi_scaled(observations)
print("Model aft trg = ", hmm.pi, hmm.A, hmm.B)

