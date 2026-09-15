import torch
import torch.nn as nn
from vectorize import get_prepared_data 
from model import TextClassifier
def main():
    print("старт")

    x,y = get_prepared_data()

    in_features = x.shape[1]
    num_class = int(y.max()) + 1

    model = TextClassifier(in_features=in_features, num_classes=num_class)
    print("\nСтруктура")
    print(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)

    logits = model(x)

    loss = loss_fn(logits,y)
    print(loss.item())

    optimizer.zero_grad()

    loss.backward()


    optimizer.step()

    new_logist = model(x)
    new_loss = loss(new_logist,y)

    print(new_loss.item())

    if new_loss.item() < loss.item():
        print("все вышло")

if __name__ == "__main__":
    main()