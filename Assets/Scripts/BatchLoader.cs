using System;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;
using Object = UnityEngine.Object;

namespace Assets.Scripts
{
	public class BatchLoader {
		// Update is called once per frame
		public static void SaveCubeMap(Cubemap cubemap, String mapId)
		{
			var tex = new Texture2D(cubemap.width, cubemap.height, TextureFormat.RGBA32, false);
			Directory.CreateDirectory($"{Application.dataPath}/training/{mapId}/envmap");
			foreach(CubemapFace face in Enum.GetValues(typeof(CubemapFace)).Cast<CubemapFace>())
			{
				if (face == CubemapFace.Unknown )continue;
				tex.SetPixels(cubemap.GetPixels(face));
				var bytes = tex.EncodeToPNG();
				File.WriteAllBytes($"{Application.dataPath}/training/{mapId}/envmap/{face.ToString()}.png", bytes);
			}
			
			
		}
		
		public static Cubemap RenderCubemap(Vector3 point)
		{
			GameObject renderCube = new GameObject("RenderCube");
			Camera renderCamera = renderCube.AddComponent<Camera>();
			renderCube.transform.position = point;
			renderCube.transform.rotation = Quaternion.identity;
			Cubemap cubemap = new Cubemap(256, TextureFormat.RGBA32, false);
			renderCamera.nearClipPlane = 0.001f;
			renderCamera.farClipPlane = 20000f;
			renderCamera.RenderToCubemap(cubemap);
			Object.Destroy(renderCube);
			return cubemap;
		}
	}
}
