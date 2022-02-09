period = 30
epochs = 150

for epoch in range(epochs):
    if (period - (epoch % period) < period*0.2) and (epoch % int(period*0.2/3) == 0):
        print(epoch)

print(1)