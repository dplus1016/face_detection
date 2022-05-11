# face_detection

[기본 정보]

1번~5번 파일은 openCV 패키지 활용

7번~10번 파일은 mediapipe 패키지 활용

- opencv 설치: pip install opencv-python
- mediapipe 설치: pip install mediapipe

---

[참고]

8번~10번 파일은 mediapipe 패키지 파일 수정 필요

- mediapipe 패키지 - python - solutions - drawing_utils.py - draw_detection 함수 끝에 아래의 내용 추가
- (추가 코드) return rect_start_point, rect_end_point
  
- 아래 이미지 참조

![image](https://user-images.githubusercontent.com/24561701/166693510-1815778d-0a0d-43a8-8b05-0748f98905ff.png)
