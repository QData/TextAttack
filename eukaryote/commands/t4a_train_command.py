import eukaryote.t4a.shared as shared
import eukaryote.t4a.train_support as train

from eukaryote.commands import TextAttackCommand


class T4A_TrainCommand(TextAttackCommand):
    def run(self, args):
        model_wrapper = shared.load_model_wrapper(args)
        dataset_train = shared.load_dataset(args)
        if args.dataset_split_eval:
            dataset_eval = shared.load_dataset(args, split=args.dataset_split_eval)
        if args.attack:
            attack = shared.load_attack(args)["attack_recipe"].build(model_wrapper)

        # Choose training function and run
        if args.model_huggingface:
            train_fn = train.train_huggingface
        elif args.model_tensorflow:
            train_fn = train.train_tensorflow

        train_fn(
            model_wrapper,
            dataset_train,
            dataset_eval=dataset_eval if args.dataset_split_eval else None,
            attack=attack if args.attack else None,
            epochs=args.epochs,
            early_stopping_epochs=args.early_stopping_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )

        # Save model
        if args.save:
            if args.model_huggingface:
                model_wrapper.model.save_pretrained(args.save)
            elif args.model_tensorflow:
                # TODO: Add saving TensorFlow models
                raise NotImplementedError

    @staticmethod
    def register_subcommand(subparsers):
        # Add parser for training
        parser_train = subparsers.add_parser("t4a_train", description="Train a model")
        shared.add_arguments_model(parser_train)
        shared.add_arguments_dataset(parser_train, default_split="train")
        shared.add_arguments_train(parser_train, default_split_eval="test")
        parser_train.set_defaults(func=T4A_TrainCommand(), attack=False)
