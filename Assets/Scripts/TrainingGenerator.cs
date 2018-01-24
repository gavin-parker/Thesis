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
        public int BatchCount;
        public int BatchSize;
        public Camera RenderCamera;
        public Collider TargetBounds;
        public Collider CameraBounds;
        public GameObject Probe;
        private Texture2D _texture2D;


        private bool _rendering;

        public void Update()
        {
            Debug.Log(transform.position);
        }

        public void Start()
        {
            String[] args = Environment.GetCommandLineArgs();
            foreach(String arg in args)
            {
                if (arg.Contains("-batch-size="))
                {
                    BatchSize = int.Parse(arg.Split('=')[1]);
                }

                if (arg.Contains("-batch-count="))
                {
                    BatchCount = int.Parse(arg.Split('=')[1]);
                }
            }
            
            _texture2D = new Texture2D(RenderCamera.pixelWidth, RenderCamera.pixelHeight, TextureFormat.RGB24, false);
            StartCoroutine(GenerateBatches());
        }

        /*
         * Generates training batches, randomising the target & light position
         */
        IEnumerator GenerateBatches()
        {
            for (int i = 0; i < BatchCount; i++)
            {
                yield return new WaitUntil(() => _rendering == false);
                GenerateBatch(RandomPointInBounds(TargetBounds.bounds), BatchSize);
            }
            Application.Quit();
        }

        /*
         * Generate a set of camera renderings for a target point, along with the environment map.
         */
        private void GenerateBatch(Vector3 targetPoint, int count)
        {
            Probe.transform.position = targetPoint;
            var cubemap = BatchLoader.RenderCubemap(targetPoint);
            var mapId = $"{SceneManager.GetActiveScene().name}_{targetPoint}";
            BatchLoader.SaveCubeMap(cubemap, mapId);
            _rendering = true;
            StartCoroutine(RenderRandomCameraAngles(targetPoint, count, mapId));
        }

        IEnumerator RenderRandomCameraAngles(Vector3 target, int count, String mapId)
        {
            Directory.CreateDirectory($"{Application.dataPath}/training/{mapId}/renders");
            for (var i = 0; i < count; i++)
            {
                Vector3 relativeCameraPos = TransformToRandomViewPoint(RenderCamera, target);
                yield return new WaitForEndOfFrame();
                RenderCamera.Render();
                _texture2D.ReadPixels(new Rect(0, 0, _texture2D.width, _texture2D.height), 0, 0);
                _texture2D.Apply();
                var bytes = _texture2D.EncodeToPNG();

                var renderId = $"{relativeCameraPos}";
                File.WriteAllBytes($"{Application.dataPath}/training/{mapId}/renders/{renderId}.png", bytes);
            }

            _rendering = false;
        }

        Vector3 TransformToRandomViewPoint(Camera renderCamera, Vector3 target)
        {
            Vector3 position = target + RandomPointInBounds(CameraBounds.bounds);
            renderCamera.transform.position = position;
            renderCamera.transform.LookAt(target);
            Vector3 relativeCameraPos = target - RenderCamera.transform.position;
            return relativeCameraPos;
        }
        
        private Vector3 RandomPointInBounds(Bounds bounds)
        {
            return new Vector3(Random.Range(bounds.min.x, bounds.max.x),
                Random.Range(bounds.min.y, bounds.max.y),
                Random.Range(bounds.min.z, bounds.max.z));
        }
    }
}