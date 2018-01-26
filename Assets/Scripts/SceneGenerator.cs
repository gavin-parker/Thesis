using System.Collections.Generic;
using UnityEngine;

namespace Assets.Scripts
{
    public class SceneGenerator
    {
        private Bounds _spawnBounds;
        private List<GameObject> _spawnedObjects;
        private Bounds _targetBounds;
        public SceneGenerator(Bounds spawnBounds, Bounds targetBounds, int objects)
        {
            _spawnBounds = spawnBounds;
            _spawnedObjects = new List<GameObject>();
            _targetBounds = targetBounds;
            GenerateScene(objects);
        }

        void CleanUp()
        {
            foreach (GameObject g in _spawnedObjects)
            {
                Object.Destroy(g);
            }
            _spawnedObjects = new List<GameObject>();
        }


        void GenerateScene(int objectCount)
        {
            CleanUp();
            for (int i = 0; i < objectCount; i++)
            {
                GenerateObject();
            }
        }


        /*
         * Random primitive with random colour
         */
        void GenerateObject()
        {
            var obj = GameObject.CreatePrimitive(RandomPrimitive());
            obj.transform.position = TrainingGenerator.RandomPointInBounds(_spawnBounds, _targetBounds);
            obj.transform.localScale = TrainingGenerator.RandomScale(Vector3.one, Vector3.one * 8);
            obj.transform.position += Vector3.up * obj.transform.localScale.y / 2;
            obj.GetComponent<Renderer>().material = RandomMaterial();
            _spawnedObjects.Add(obj);
        }

        private Material RandomMaterial()
        {
            Material material = new Material(Shader.Find("Standard"));
            material.color = Random.ColorHSV(0, 1);
            return material;
        }

        private static PrimitiveType RandomPrimitive()
        {
            switch (Random.Range(0, 4))
            {
                case 1:
                    return PrimitiveType.Sphere;
                case 2:
                    return PrimitiveType.Cylinder;
                case 3:
                    return PrimitiveType.Capsule;
                default:
                    return PrimitiveType.Cube;
            }
        }
    }
}