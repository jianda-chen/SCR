import distracting_control

class DCSuite:
    def __init__(
            self,
            domain_name,
            task_name,
            difficulty,
            resource_files,
            seed,
            from_pixels,
            height,
            width,
            frame_skip,
            ) -> None:
        self.env = distracting_control.make('distracting_control:Walker-walk-easy-v1')

def make(domain_name,
        task_name,
        difficulty=None,
        intensity=None,
        distraction_types='background',
        sample_from_edge=False,
        channels_first=True,
        width=84,
        height=84,
        frame_skip=4,
        from_pixels=True,
        seed=None,
        max_episode_steps=1000,
    ):
    return distracting_control.make_env(
                from_pixels=from_pixels,
                frame_skip=frame_skip,
                max_episode_steps=max_episode_steps,
                domain_name=domain_name,
                task_name=task_name,
                difficulty=difficulty,
                intensity=intensity,
                sample_from_edge=sample_from_edge,
                channels_first=channels_first,
                width=width,
                height=height,
                distraction_seed=seed,
                distraction_types=('background', 'camera', 'color'),
                background_data_path='../DAVIS2017/DAVIS/JPEGImages/480p',
                # dynamic=True,
                )


