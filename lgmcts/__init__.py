def make(
    task_name: str | None,
    *,
    task_kwargs: dict | None = None,
    modalities: Literal["rgb", "depth", "segm"]
    | list[Literal["rgb", "depth", "segm"]]
    | None = None,
    seed: int | None = None,
    debug: bool = False,
    display_debug_window: bool = False,
    render_prompt: bool = False,
    record_gui: bool = False,
    record_kwargs: dict | None = None,
    hide_arm_rgb: bool = True,
) -> VIMAEnvBase:
    if record_gui:
        record_kwargs = record_kwargs or dict(video_name="gui_record.mp4")
        env = GUIRecorder(
            modalities=modalities,
            task=task_name,
            task_kwargs=task_kwargs,
            seed=seed,
            debug=debug,
            display_debug_window=display_debug_window,
            hide_arm_rgb=hide_arm_rgb,
            **record_kwargs,
        )
    else:
        env = VIMAEnvBase(
            modalities=modalities,
            task=task_name,
            task_kwargs=task_kwargs,
            seed=seed,
            debug=debug,
            display_debug_window=display_debug_window,
            hide_arm_rgb=hide_arm_rgb,
        )
    if render_prompt:
        env = PromptRenderer(env)
    return env