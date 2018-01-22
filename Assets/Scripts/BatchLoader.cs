using System;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace Assets.Scripts
{
	public class BatchLoader : MonoBehaviour {
		// Use this for initialization

		void Start ()
		{
			Cubemap testMap = RenderCubemap(new Vector3(0, 0, 0));
			SaveCubeMap(testMap, "test");
		}

		void Update () {
		
		}

		// Update is called once per frame
		void SaveCubeMap(Cubemap cubemap, String mapId)
		{
			var tex = new Texture2D(cubemap.width, cubemap.height, TextureFormat.RGBA32, false);
			Directory.CreateDirectory($"{Application.dataPath}/maps/{mapId}");
			foreach(CubemapFace face in Enum.GetValues(typeof(CubemapFace)).Cast<CubemapFace>())
			{
				if (face == CubemapFace.Unknown )continue;
				Debug.Log(face);
				tex.SetPixels(cubemap.GetPixels(face));
				var bytes = tex.EncodeToPNG();
				File.WriteAllBytes($"{Application.dataPath}/maps/{mapId}/{face.ToString()}.png", bytes);
			}
			
			
		}
		
		Cubemap RenderCubemap(Vector3 point)
		{
			GameObject renderCube = new GameObject("RenderCube");
			renderCube.AddComponent<Camera>();
			renderCube.transform.position = point;
			renderCube.transform.rotation = Quaternion.identity;
			Cubemap cubemap = new Cubemap(256, TextureFormat.RGBA32, false);
			renderCube.GetComponent<Camera>().RenderToCubemap(cubemap);
			
			return cubemap;
		}
	}
}
