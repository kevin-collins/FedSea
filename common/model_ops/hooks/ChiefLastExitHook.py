from tensorflow.python.training import session_run_hook

class ChiefLastExitHook(session_run_hook.SessionRunHook):
    """Make sure chief exit after worker (except evaluator),
       After training finish, chief has to do evaluation and
       model export, so chief exits later than workers.
    """

    def __init__(self, num_worker, is_chief):
        """
        Args:
           num_worker: workers + chief number (except evaluator)
           is_chief: is chief
        """
        self._num_worker = num_worker
        self._is_chief = is_chief
        self._queue = None
        self._queue_size = None
        self._enque_op = None

    def begin(self):
        """Count the number of workers and chief, and setup exit counter queue
        """
        logging.info('number workers (including chief) = %d' % self._num_worker)
        with tf.device(tf.DeviceSpec(job='ps', task=0, device_type='CPU', device_index=0)):
            self._queue = tf.FIFOQueue(capacity=self._num_worker, dtypes=[tf.float32], shapes=[()],
                                       name='worker_exit_counter', shared_name='worker_exit_counter')
        self._enque_op = self._queue.enqueue(1.0)
        self._queue_size = self._queue.size()

    def after_create_session(self, session, coord):
        """Clean up the queue, as there are sometimes ps is not
           exit, the last run enqueued elements will remain in the queue
        """
        if self._is_chief:
            queue_size = session.run(self._queue_size)
            logging.info('create session, ChiefLastExitHook initial queue size: %d' % queue_size)

    def end(self, session):
        """Only when all workers and chief enqueue an element, will the end method exit
        """
        if not self._is_chief:
            session.run(self._enque_op)
        if self._is_chief:
            queue_size = session.run(self._queue_size)
            logging.info('queue size = %d' % queue_size)
            while queue_size < self._num_worker - 1:
                queue_size = session.run(self._queue_size)
                time.sleep(5)
                logging.info('waiting for other worker to exit, finished %d, total %d' % (queue_size, self._num_worker))
        logging.info('ChiefLastExitHook done')

