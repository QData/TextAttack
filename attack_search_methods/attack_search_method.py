

class AttackSearchMethod:
    """ An abstract class for attack search methods to implement.
        
        Each attack() method should take a list of (label, sentence) pairs
        and return a list of AttackResults.
    """
    
    def attack(self):
        raise NotImplementedException()