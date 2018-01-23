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

        private RenderTexture _renderTexture;

        public void Start()
        {
            _renderTexture = new RenderTexture(RenderCamera.pixelWidth,RenderCamera.pixelHeight,16, RenderTextureFormat.ARGB32);
            _renderTexture.Create();
            RenderCamera.targetTexture = _renderTexture;
            RenderTexture.active = _renderTexture;
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
            var tex = new Texture2D(_renderTexture.width, _renderTexture.height, TextureFormat.RGBA32, false);
            Directory.CreateDirectory($"{Application.dataPath}/training/{mapId}/renders");
            while (!_renderTexture.IsCreated())
            {
                yield return new WaitForEndOfFrame();
            }
            for (var i = 0; i < count; i++)
            {
                TransformToRandomViewPoint(RenderCamera, target, 2f, 10f);
                yield return new WaitForEndOfFrame();
                Debug.Log(RenderCamera.transform.position);
                tex.ReadPixels(new Rect(0,0, _renderTexture.width, _renderTexture.height), 0, 0);
                tex.Apply();
                var bytes = tex.EncodeToPNG();
                Vector3 relativeCameraPos = RenderCamera.transform.position - target;
                var renderId = $"{relativeCameraPos}";
                File.WriteAllBytes($"{Application.dataPath}/training/{mapId}/renders/{renderId}.png", bytes);
            }
        }
        

        void TransformToRandomViewPoint(Camera renderCamera, Vector3 target, float maxDistance, float minDistance)
        {
            float dist = Random.Range(minDistance, maxDistance);
            Vector3 position = target + Random.onUnitSphere*dist;
            Debug.Log($"New position is {position}");
            renderCamera.transform.position = position;
            renderCamera.transform.LookAt(target - renderCamera.transform.position);
        }
        
    }
}