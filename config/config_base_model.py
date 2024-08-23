import datetime

class model_config_builder_svg():
    def __init__(self, config):
        self.lr                          = config.learning_rate
        self.beta                        = config.beta
        self.beta1                       = config.beta1
        self.batch_size                  = config.batch_size
        self.optimizer                   = config.optimizer
        self.criterion                   = config.criterion
        self.device                      = config.device
        self.g_dim                       = config.g_dim
        self.z_dim                       = config.z_dim
        self.state_action_size           = config.state_action_size
        self.rnn_size                    = config.rnn_size
        self.predictor_rnn_layers        = config.predictor_rnn_layers
        self.posterior_rnn_layers        = config.posterior_rnn_layers
        self.prior_rnn_layers            = config.prior_rnn_layers
        self.channels                    = config.channels
        self.model_dir                   = config.model_dir
        self.model_name                  = config.model_name
        self.model_name_save_appendix    = config.model_name_save_appendix
        self.n_past                      = config.n_past
        self.n_future                    = config.n_future
        self.n_eval                      = config.n_eval
        self.tactile_size                = config.tactile_size

class Config:
    # add model config
    def __init__(self):
        ###########################
        # setup parameters
        ###########################
        
        self.debug == True

        ###########################
        # Training parameters
        ###########################
        self.seed       = 42
        self.batch_size = 256

        self.num_steps       = 25_000          # dataset is currently 144,495 steps at 256 batch size is:  560ish steps per epoch
        self.save_interval   = 10_000
        self.log_interval    = 100
        if self.debug: self.eval_interval   = 10
        else:          self.eval_interval   = 500 # 500

        self.sample_rate = 2                  # how many frames to skip for the dataset (basically makes bigger changes in between each sequence) 

        self.num_frames 	        = 20     # just context length + 1 ( + 1 because its the prediction horizon for autoregressive models)
        self.context_length       = 10
        self.prediction_horizon   = 100    # when rolling out autoregressive models, this is the prediction horizon for testing (not training)

        self.num_workers = 4
        self.device = "cuda"

        self.shuffle_buffer_size     = 1000
        self.val_shuffle_buffer_size = 1000

        ###########################
        # optimizer parameters
        ###########################
        self.criterion = "MAE"

        self.beta1 	       = 0.9
        self.beta2 	       = 0.99
        self.weight_decay  = 1e-4 
        self.learning_rate = 0.001

        ###########################
        # SPOTS-SVG parameters
        ###########################
        self.beta                      = 0.0001
        self.optimizer                 = 'adam'

        self.n_past                    = self.context_length
        self.n_future                  = self.num_frames - self.context_length
        self.n_eval                    = self.prediction_horizon
        
        self.channels                  = self.input_dim
        self.out_channels              = self.input_dim
        self.model_dir                 = ""
        self.model_name_save_appendix  = ""

        self.state_action_size         = 6 * 2
        self.rnn_size                  = 256 # Large: 512 , Medium: 256
        self.predictor_rnn_layers      = 6   # Large: 6   , Medium: 4
        self.posterior_rnn_layers      = 4   # Large: 4   , Medium: 3
        self.prior_rnn_layers          = 4   # Large: 4   , Medium: 3
        self.g_dim                     = 256 # Large: 512 , Medium: 256
        self.z_dim                     = 10
        self.tactile_size              = 48

        ###########################
        # build sub-configs
        ###########################
        self.model_config = model_config_builder_svg(self)

    def to_dict(self):
        ''' returns a dictionary of all the self variables and nest the model_config as well
            the function must be repeatable '''
        def recursive_to_dict(obj):
            if isinstance(obj, dict):
                return {k: recursive_to_dict(v) for k, v in obj.items()}
            elif hasattr(obj, "__dict__"):
                return {k: recursive_to_dict(v) for k, v in obj.__dict__.items()}
            else:
                return obj
        
        return recursive_to_dict(self)

    def __repr__(self):
        return str(self.to_dict())
