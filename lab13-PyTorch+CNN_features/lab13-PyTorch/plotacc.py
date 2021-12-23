import matplotlib.pyplot as plt
import numpy as np
epochs = list(range(1,16))
acc = [.5349,.554,.5243,.6223,.6519,.6584,.6429,.7447,.7087,.7557,.7622,.7432,.6628,.7351,.7599]
plt.plot(epochs, acc,'o',linestyle='-')
plt.xlabel("Epochs")
plt.xlim(0,16)
plt.ylabel("Accuracy")
plt.title("")
plt.savefig("acc_epochs.pdf")
plt.show()