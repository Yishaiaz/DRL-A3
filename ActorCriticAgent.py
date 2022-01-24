from abc import ABC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,regularizers
import tensorflow_probability as tfp
import numpy as np
        

class Critic(object):
    def __init__(self,state_dim,hidden_dim,name,lr=0.001) -> None:
        super(Critic,self).__init__()
        self.name = name
        self.state = layers.Input(shape=state_dim)
        self.fc1 = layers.Dense(hidden_dim[0],activation='relu',name='fc1')(self.state)
        self.fc2 = layers.Dense(hidden_dim[1], activation='relu',name='fc2')(self.fc1)
        self.output = layers.Dense(1)(self.fc2)
        
        self.optimizer=optimizers.Adam(learning_rate=lr)
        self.model = keras.Model(self.state,self.output)
        
    def loss(self,y_true,y_pred):
        return losses.mean_squared_error(y_true,y_pred)
    
    def predict(self, state):
        return self.model(state)
    
    def save_base(self):
        base_model = keras.Model(self.state,self.fc2)
        base_model.save_weights('./weights/'+self.name+'_critic.h5')  

    def load_base(self,path):
        self.model.load_weights(path,by_name=True)    

    def save_model(self):
        self.model.save('models/'+self.name+'_critic.h5') 
    
    def load_model(self):
        self.model = keras.models.load_model('models/'+self.name+'_critic.h5')
    
         
class ActorBase(ABC):
    def __init__(self,state_dim,hidden_dim,name, activation='relu') -> None:
        super(ActorBase,self).__init__()
        self.name = name
        self.state = layers.Input(shape=state_dim,name='input')

        self.fc1 = layers.Dense(hidden_dim[0],activation=activation,name='fc1_'+self.name)(self.state)
        self.baseout = layers.Dense(hidden_dim[1],activation=activation,name='baseout_'+self.name)(self.fc1)

    
    def save_base(self):
        base_model = keras.Model(self.state,self.baseout)
        base_model.save_weights('weights/'+self.name+'_actor.h5')
    
    def load_base(self,path):
        self.model.load_weights(path,by_name=True)
        
    def save_model(self):
        self.model.save('models/'+self.name+'_actor.h5')

    def load_model(self):
        self.model = keras.models.load_model('models/'+self.name+'_actor.h5')
    
    def freeze_train(self):
        self.fc1.trainable=False
        self.baseout.trainable=False
         
    def resume_train(self):
        self.fc1.trainable=True
        self.baseout.trainable=True

class ActorSoftmax(ActorBase):
    def __init__(self, state_dim, hidden_dim, actions_dim,name, lr=0.001) -> None:
        super(ActorSoftmax,self).__init__(state_dim, hidden_dim,name)
        
        self.output = layers.Dense(actions_dim,activation='softmax')(self.baseout)
        self.model = keras.Model(self.state,self.output)
        self.optimizer=optimizers.Adam(learning_rate=lr)
        
    def loss(self,y_true,y_pred,delta):
        log_prob = (losses.categorical_crossentropy(y_true,y_pred,from_logits=False))
        return log_prob*delta
        
    
    def predict(self,state):
        return self.model(state)
    
    def take_action(self,state,current_env_action_size):
        prob = self.predict(state)[0]
        prob = prob.numpy()[:current_env_action_size]
        prob = prob/np.sum(prob)
        action = np.random.choice(range(current_env_action_size),p=prob)
        return action
    
class ActorDist(ActorBase):
    def __init__(self, state_dim, hidden_dim, min_action, max_action,name, lr=0.001) -> None:
        self.min_action = min_action
        self.max_action = max_action
        
        super(ActorDist,self).__init__(state_dim, hidden_dim,name,activation='elu')
        
        self.mu = tf.nn.tanh(layers.Dense(1,kernel_regularizer=regularizers.l1_l2(l2=1e-3))(self.baseout))
        self.sigma = tf.clip_by_value(tf.nn.softplus(layers.Dense(1,kernel_regularizer=regularizers.l1_l2(l2=1e-3))(self.baseout)),1e-5,self.max_action)
  
        self.model = keras.Model(self.state,[self.mu, self.sigma])
        self.optimizer=optimizers.Adam(learning_rate=lr)
        
    def loss(self,action,norm_dist,delta):
        log_prob = -tf.math.log(norm_dist.prob(action) + 1e-5)
        return log_prob*delta
        
    
    def predict(self,state):
        return self.model(state)
    
    def take_action(self,state):
        mu,sigma = self.predict(state)
        norm_dist = tfp.distributions.Normal(mu, sigma)
        action = tf.squeeze(norm_dist.sample(1), axis=0)
        action = tf.clip_by_value(action, self.min_action, self.max_action)
        action = tf.squeeze(action)
        return np.array([action.numpy()]), norm_dist
    
        

        