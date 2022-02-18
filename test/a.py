d = {
    "i" for i in range(3)
     }


d["a"].update({"aa":1})
d["a"].update({"bb":1})

print(d)
