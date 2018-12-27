import re
def load_state(file_name,args):
    mystate = Mystate(args)
    with open(file_name,"r") as f:
        state_dict = dict()
        for state in f:
            state=state.replace("\n","")
            state = state.split("=")
            if len(state) != 2:
                print(state)
                print("Error")
                exit()
            state[1] = state[1].replace(" ","")
            state[0] = state[0].replace(" ","")
            if state[0][0] == "#":
                continue
            if re.match("\"",state[1]) or re.match("\'",state[1]):
                state[1] = state[1].replace("\"","")
                state[1] = state[1].replace("\'","")
                state[1] = str(state[1])
            elif re.search("\.",state[1]):
                state[1] = float(state[1])
            else:
                state[1] = int(state[1])
            state_dict[state[0]] = state[1]
    mystate(state_dict)
    return mystate
class Mystate():
    def __init__(self,args):
        for key,value in args.__dict__.items():
            setattr(self,key,value)
    def __call__(self,mydict):
        for key,value in mydict.items():
            setattr(self,key,value)
