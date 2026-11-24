import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from glzn.proc import ProcDeps, Processor, load_checkpoint, save_checkpoint
from glzn.proc.step import Phase, StepTracker


class ProcessorTests(unittest.TestCase):
    def test_resume_at_update_boundary_with_short_final_window(self):
        tracker = StepTracker.init(
            runstart=0.0,
            epochstart=0.0,
            trainsamples=3,
            valsamples=0,
            microbatch_size=1,
            accum_steps=2,
            total_epochs=1,
        )
        inputs = torch.ones(1, 1)

        def make_processor():
            model = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(model.weight)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            processor = Processor(ProcDeps(model=model, optimizer=optimizer), tracker=tracker)
            return model, processor

        def train_microbatch(model, processor, now):
            with processor.batch(phase=Phase.TRAIN, now=now) as batch:
                batch.backward((model(inputs) - 1).square().mean())

        model, processor = make_processor()
        train_microbatch(model, processor, now=1.0)
        self.assertEqual(processor.s.fullstep, 0)
        train_microbatch(model, processor, now=2.0)
        self.assertEqual(processor.s.fullstep, 1)

        resumed_model, resumed = make_processor()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, state={"model": model, "processor": processor})
            load_checkpoint(path, state={"model": resumed_model, "processor": resumed})

        self.assertEqual(resumed.s, processor.s)
        train_microbatch(model, processor, now=3.0)
        train_microbatch(resumed_model, resumed, now=3.0)
        self.assertEqual(resumed.s, processor.s)
        self.assertEqual(resumed.s.fullstep, 2)
        self.assertEqual(resumed.s.microstep, 0)
        torch.testing.assert_close(model.weight, torch.tensor([[0.54]]))
        torch.testing.assert_close(resumed_model.weight, model.weight, rtol=0, atol=0)
