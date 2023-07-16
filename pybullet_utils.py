def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Loads the given URDF filepath."""
    assert (
        "physicsClientId" in kwargs
    ), "You MUST explicitly provide pybullet client ID for safety reason!"
    # Handles most general file open case.
    try:
        return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet_client.error:
        pass
