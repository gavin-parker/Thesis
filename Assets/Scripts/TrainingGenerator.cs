using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;
using Random = UnityEngine.Random;

namespace Assets.Scripts
{
    public class TrainingGenerator : MonoBehaviour
    {
        public Camera RenderCamera;
        private Texture2D _texture2D;
        
        public void Update()
        {
            Debug.Log(transform.position);
        }
        
        public void Start()
        {
            _texture2D = new Texture2D(RenderCamera.pixelWidth, RenderCamera.pixelHeight, TextureFormat.RGB24, false);
            GenerateBatch(new Vector3(0,0,0), 5);
        }
        /*
         * Generate a set of camera renderings for a target point, along with the environment map.
         */
        public void GenerateBatch(Vector3 targetPoint, int count)
        {
            var cubemap = BatchLoader.RenderCubemap(targetPoint);
            var mapId = $"{SceneManager.GetActiveScene().name}_{targetPoint}";
            BatchLoader.SaveCubeMap(cubemap, mapId);
            StartCoroutine(RenderRandomCameraAngles(targetPoint, count, mapId));
        }

        IEnumerator RenderRandomCameraAngles(Vector3 target, int count, String mapId)
        {
            Directory.CreateDirectory($"{Application.dataPath}/training/{mapId}/renders");
            for (var i = 0; i < count; i++)
            {
                Vector3 relativeCameraPos = TransformToRandomViewPoint(RenderCamera, target, 20f, 60f);
                yield return new WaitForEndOfFrame();
                RenderCamera.Render();
                _texture2D.ReadPixels(new Rect(0,0, _texture2D.width, _texture2D.height), 0, 0);
                _texture2D.Apply();
                var bytes = _texture2D.EncodeToPNG();
                
                var renderId = $"{relativeCameraPos}";
                File.WriteAllBytes($"{Application.dataPath}/training/{mapId}/renders/{renderId}.png", bytes);
            }
            Application.Quit();
        }
        
        Vector3 TransformToRandomViewPoint(Camera renderCamera, Vector3 target, float maxDistance, float minDistance)
        {
            float dist = Random.Range(minDistance, maxDistance);
            Vector3 position = target + Random.onUnitSphere*dist;
            Debug.Log($"New position is {position}");
            renderCamera.transform.position = position;
            renderCamera.transform.LookAt(target);
            Vector3 relativeCameraPos = target - RenderCamera.transform.position;
            return relativeCameraPos;
        }
        
    }
}