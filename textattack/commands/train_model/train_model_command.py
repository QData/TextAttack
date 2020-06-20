from argparse import ArgumentParser

from textattack.commands import TextAttackCommand


class TrainModelCommand(TextAttackCommand):
    """
    The TextAttack train module:
    
        A command line parser to train a model from user specifications.
    """

    def run(self, args):
        
        if not args.model_dir:
            args.model_dir = 'bert-base-cased' if args.cased else 'bert-base-uncased'
        
        if args.output_prefix: args.output_prefix += '-'
        
        cased_str = '-' + ('cased' if args.cased else 'uncased')
        date_now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        root_output_dir = 'outputs'
        args.output_dir = os.path.join(root_output_dir, 
            f'{args.output_prefix}{args.dataset}{cased_str}-{date_now}/')
        
        # Use multiple GPUs if we can!
        args.num_gpus = torch.cuda.device_count()
        
        raise NotImplementedError("Cannot train models yet - stay tuned!!")

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser("train", help="train a model")parser.add_argument('--model_dir', type=str, default=None, 
        help='directory of model to train')
    
        parser.add_argument('--logging_steps', type=int, 
            default=500, help='log model after this many steps')
        
        parser.add_argument('--checkpoint_steps', type=int, 
            default=5000, help='save model after this many steps')
        
        parser.add_argument('--checkpoint_every_epoch', action='store_true',
            default=False, help='save model checkpoint after this many steps')
        
        parser.add_argument('--output_prefix', type=str,
            default='', help='prefix for model saved output')
        
        parser.add_argument('--dataset', type=str,
            default='yelp', help='dataset for training, like \'yelp\'')
        
        parser.add_argument('--cased', action='store_true', default=False,
             help='if true, bert is cased, if false, bert is uncased')
        
        parser.add_argument('--debug_cuda_memory', type=str,
            default=False, help='Print CUDA memory info periodically')
        
        parser.add_argument('--num_train_epochs', '--epochs', type=int, 
            default=100, help='Total number of epochs to train for')
            
        parser.add_argument('--early_stopping_epochs', type=int, 
            default=-1, help='Number of epochs validation must increase'
                               ' before stopping early')
            
        parser.add_argument('--batch_size', type=int, default=128, 
            help='Batch size for training')
            
        parser.add_argument('--max_seq_len', type=int, default=128, 
            help='Maximum length of a sequence (anything beyond this will '
                 'be truncated) - '
                 '# BERT\'s max seq length is 512 so can\'t go higher than that.')
            
        parser.add_argument('--learning_rate', '--lr', type=int, default=2e-5, 
            help='Learning rate for Adam Optimization')
            
        parser.add_argument('--tb_writer_step', type=int, default=1000, 
            help='Number of steps before writing to tensorboard')
            
        parser.add_argument('--grad_accum_steps', type=int, default=1, 
            help='Number of steps to accumulate gradients before optimizing, '
                    'advancing scheduler, etc.')
            
        parser.add_argument('--warmup_proportion', type=int, default=0.1, 
            help='Warmup proportion for linear scheduling')
            
        parser.add_argument('--config_name', type=str, default='config.json', 
            help='Filename to save BERT config as')
            
        parser.add_argument('--weights_name', type=str, default='pytorch_model.bin', 
            help='Filename to save model weights as')
        
        args = parser.parse_args()
