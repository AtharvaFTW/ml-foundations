import math

class Value:
    def __init__(self, data,_children =()):
        self.data = data
        self._children= set(_children)
        self.grad = 0.0
        self._backward = lambda: None


    def __repr__(self):
        return (f"Value {self.data}")
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
    
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t,(self,))

        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward =_backward
        return out

    def __add__(self,other):
        out = Value(self.data + other.data , (self , other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self,other):
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    
    
if __name__ =="__main__":
    
    a = Value(0.5)
    b = Value(0.5)
    c = (a*b).tanh()
    c.backward()
    print("a grad :",a.grad)
    print("b grad :",b.grad )
    