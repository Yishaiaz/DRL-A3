from abc import ABC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,regularizers
import tensorflow_probability as tfp
import numpy as np

class ProgressiveActorBase(ABC):
    
    def __init__(self,model_a,model_b,state_dim,hidden_dim,name,activation=layers.ReLU()) -> None:
        super(ProgressiveActorBase,self).__init__()
        self.name = name
        
        model_a.model.trainable=False
        model_b.model.trainable=False
        
        self.state = layers.Input(shape=state_dim,name='input')

        self.fc1 = layers.Dense(hidden_dim[0],activation='relu',name='fc1_p')(self.state)
        self.ad1_a = layers.Dense(hidden_dim[0],activation='relu')(model_a.model.get_layer('fc1_'+model_a.name)(self.state))
        self.ad1_b = layers.Dense(hidden_dim[0],activation='relu')(model_b.model.get_layer('fc1_'+model_b.name)(self.state))
        self.layer_one = activation(self.fc1+self.ad1_a+self.ad1_b)
        
        self.fc2 = layers.Dense(hidden_dim[1],activation='relu',name='fc2_p')(self.layer_one)
        self.ad2_a = layers.Dense(hidden_dim[1],activation='relu')(model_a.model.get_layer('baseout_'+model_a.name)(model_a.model.get_layer('fc1_'+model_a.name)(self.state)))
        self.ad2_b = layers.Dense(hidden_dim[1],activation='relu')(model_b.model.get_layer('baseout_'+model_b.name)(model_b.model.get_layer('fc1_'+model_b.name)(self.state)))
        self.baseout = activation(self.fc2+self.ad2_a+self.ad2_b)
        

    def save_model(self):
        self.model.save('models/'+self.name+'_progressive_actor.h5')

    def load_model(self):
        self.model = keras.models.load_model('models/'+self.name+'_progressive_actor.h5')

class ProgressiveActorSoftmax(ProgressiveActorBase):
    def __init__(self, model_a, model_b,state_dim, hidden_dim, actions_dim,name, lr=0.001) -> None:
        super(ProgressiveActorSoftmax,self).__init__( model_a, model_b,state_dim, hidden_dim,name)
        
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
    
class ProgressiveActorDist(ProgressiveActorBase):
    def __init__(self,  model_a, model_b, state_dim, hidden_dim, min_action, max_action,name, lr=0.001) -> None:
        self.min_action = min_action
        self.max_action = max_action
        
        super(ProgressiveActorDist,self).__init__(model_a, model_b,state_dim, hidden_dim,name,activation=layers.ELU())
        
        self.mu = tf.nn.tanh(layers.Dense(1,kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3))(self.baseout))
        self.sigma = tf.clip_by_value(tf.nn.softplus(layers.Dense(1,kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3))(self.baseout)),1e-5,self.max_action)
  
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
            