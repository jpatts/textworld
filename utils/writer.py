import tensorflow as tf
import os
from datetime import datetime
from pathlib import Path


class Writer(object):

    def __init__(self, extension, save_dir, log_dir, log_freq):
        super(Writer, self).__init__()
        self.global_step = tf.train.get_or_create_global_step()

        if not Path(save_dir).is_dir():
            os.mkdir(save_dir)
        if extension is '':
            extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.log_path = log_dir + extension
        self.writer = tf.contrib.summary.create_file_writer(self.log_path)
        self.writer.set_as_default()
        self.log_freq = log_freq

        self.save_path = save_dir + extension
        self.ckpt_prefix = self.save_path + '/ckpt'
    
    def save(self, model, optim, epoch):
        saver = tf.train.Checkpoint(model=model, optim=optim, epoch=epoch, global_step=self.global_step)
        saver.save(self.ckpt_prefix)

    def restore(self, model, optim, epoch):
        saver = tf.train.Checkpoint(model=model, optim=optim, epoch=epoch, global_step=self.global_step)
        status = saver.restore(tf.train.latest_checkpoint(self.save_path))
        status.assert_existing_objects_matched()
        status.assert_consumed()
        return model, optim, epoch

    def log(self, optim, tape, loss):
        if self.global_step.numpy() % self.log_freq == 0:
            with tf.contrib.summary.always_record_summaries():
                # Log vars
                status = tf.contrib.summary.scalar('loss', loss)
                
                if self.global_step.numpy() % (self.log_freq * 10) == 0:
                    # Log weights
                    slots = optim.get_slot_names()
                    for variable in tape.watched_variables():
                            tf.contrib.summary.histogram(variable.name, variable)
                            for slot in slots:
                                slotvar = optim.get_slot(variable, slot)
                                if slotvar is not None:
                                    tf.contrib.summary.histogram(variable.name + '/' + slot, slotvar)

    def log_state(self, name, state):
        if self.global_step.numpy() % (self.log_freq * 10) == 0:
            with tf.contrib.summary.always_record_summaries():
                state = tf.cast(state, tf.float32)
                tf.contrib.summary.image(name, state, max_images=3)
    
    def log_var(self, name, var):
        if self.global_step.numpy() % self.log_freq == 0:
            with tf.contrib.summary.always_record_summaries():
                status = tf.contrib.summary.scalar(name, var)
