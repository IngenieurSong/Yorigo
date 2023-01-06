from scenedetect import open_video, ContentDetector, SceneManager, StatsManager, scene_manager

# 영상의 내용이 일정 threshold 이상 바뀌면 다른 scene으로 감지
def find_scenes(video_path):
    video_stream = open_video(video_path)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    scene_manager.add_detector(ContentDetector())

    scene_manager.detect_scenes(video=video_stream)
    scene_list = scene_manager.get_scene_list()

    return scene_list

# 감지된 scene의 시작 시간과 종료 시간을 리스트로 저장
def get_time_list(scenes):
    scene_time_list = []

    for scene in scenes:
        scene_time_list.append((scene[0].get_timecode(), scene[1].get_timecode()))

    return scene_time_list

# 감지된 scene의 프레임 저장
def save_scene_frame(video_path, save_dir):
    scenes = find_scenes(video_path)
    # print(scenes)
    video_stream = open_video(video_path)
    scene_manager.save_images(scenes, video_stream, 1, output_dir = save_dir)

    scene_time_list = get_time_list(scenes)

    return scene_time_list