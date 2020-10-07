# small class to store atomic data
class Element():
    def __init__(self,atom):
        self.atom = atom
        self.final_states = []
        self.E = []
        self.ker = []
        self.rates = {}
        self.br = {}
        
        self.temp = [1000,3000,5000,7000,9000]
        for i in range(len(self.temp)):
            self.rates[self.temp[i]] = []
            self.br[self.temp[i]] = []

    def read_data_line(self,line):
        line_s = line.split()
        # print(line_s,len(line_s))
        self.final_states.append(line_s[1])
        self.E.append(float(line_s[2]))
        self.ker.append(float(line_s[3]))

        for T,i in zip(self.temp,range(4,4+len(self.temp)*2,2)):
            self.rates[T].append(float(line_s[i]))
            self.br[T].append(float(line_s[i+1]))

    def set_mass(self,mass):
        self.mass = mass

    def __repr__(self):
        return self.atom


# def read_Jon_data(filepath):
#     '''
#     Reads Jons atomic data into a dictionary with name of atom as key
#     and values are Element objects
#     '''
#     data = {}

#     with open(filepath,'r') as f:
#         for i in range(25):
#             l = f.readline()

#         for i in range(13):
#             l = f.readline()
#             l = l.split('|')
#             name = l[1].split()[0]
#             data[name] = Element(name)

#         reading = False
#         for line in f:
#             if not reading:
#                 line = line.split('+')
#                 if line[0].strip() in data.keys() and line[1][0] == 'H':
#                     atom = line[0].strip()
#                     reading = True

#             else:
#                 if line == '\n':
#                     reading=False
#                     continue

#                 line_s = line.split()

#                 if line_s[0] == '----->':
#                     data[atom].read_data_line(line)

#     return data

def read_Jon_data(filepath):
    '''
    Reads Jons atomic data into a dictionary with name of atom as key
    and values are Element objects
    '''
    data = {}

    with open(filepath,'r') as f:
        
        reading = False
        for line in f:
            if not reading:
                # print(line)
                line = line.split('+')

                try:
                    el2 = line[1][0]
                except IndexError:
                    continue

                if el2 == 'H':
                    atom = line[0].strip()
                    reading = True
                    data[atom] = Element(atom)

            else:
                if line == '\n':
                    reading=False
                    continue

                line_s = line.split()

                if line_s[0] == '----->':
                    data[atom].read_data_line(line)

    return data