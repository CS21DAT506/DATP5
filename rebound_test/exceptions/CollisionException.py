class CollisionException(Exception):
    def __init__(self, message, agent):
        super().__init__(message)
        self.agent = agent