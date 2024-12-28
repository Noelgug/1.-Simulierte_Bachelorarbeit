from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def get_params(self):
        """Get model parameters"""
        pass
