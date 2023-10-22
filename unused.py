    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Particle(self.states+other.states, self.directions+other.directions)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'").format(self.__class__, type(other))
        
    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return Particle(self.states-other.states, self.directions-other.directions)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'").format(self.__class__, type(other))
        
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return Particle(self.states*other.states, self.directions*other.directions)
        if isinstance(other, int) or isinstance(other, float):
            return Particle(self.states*other, self.directions*other)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'").format(self.__class__, type(other))
    
    def __rmul__(self, other):
        if isinstance(other, self.__class__):
            return Particle(self.states*other.states, self.directions*other.directions)
        if isinstance(other, int) or isinstance(other, float):
            return Particle(self.states*other, self.directions*other)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'").format(self.__class__, type(other))
        
    def __div__(self, other):
        if isinstance(other, self.__class__):
            return Particle(self.states/other.states, self.directions/other.directions)
        if isinstance(other, int) or isinstance(other, float):
            return Particle(self.states/other, self.directions/other)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'").format(self.__class__, type(other))