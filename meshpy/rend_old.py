    def images(self, mesh, object_to_camera_poses,
               mat_props=None, light_props=None, enable_lighting=True, debug=False):
        """Render images of the given mesh at the list of object to camera poses.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The mesh to be rendered.
        object_to_camera_poses : :obj:`list` of :obj:`RigidTransform`
            A list of object to camera transforms to render from.
        mat_props : :obj:`MaterialProperties`
            Material properties for the mesh
        light_props : :obj:`MaterialProperties`
            Lighting properties for the scene
        enable_lighting : bool 
            Whether or not to enable lighting
        debug : bool
            Whether or not to debug the C++ meshrendering code.

        Returns
        -------
        :obj:`tuple` of `numpy.ndarray`
            A 2-tuple of ndarrays. The first, which represents the color image,
            contains ints (0 to 255) and is of shape (height, width, 3). 
            Each pixel is a 3-ndarray (red, green, blue) associated with a given
            y and x value. The second, which represents the depth image,
            contains floats and is of shape (height, width). Each pixel is a
            single float that represents the depth of the image.
        """
        #tracemalloc.start()
        # get mesh spec as numpy arrays
        #vertex_arr = mesh.vertices
        #tri_arr = mesh.triangles.astype(np.int32)
        #if mesh.normals is None:
        #    mesh.compute_vertex_normals()
        #norms_arr = mesh.normals

        # set default material properties
        #if mat_props is None:
        #    mat_props = MaterialProperties()
        #mat_props_arr = mat_props.arr

        # set default light properties
        #if light_props is None:
        #    light_props = LightingProperties()

        # render for each object to camera pose
        # TODO: clean up interface, use modelview matrix!!!!
        color_ims = []
        depth_ims = []
        render_start = time.time()


        #mesh_rend = pyrender.Mesh.from_trimesh(mesh.trimesh)
        #Visualizer3D.figure()
        #scene = pyrender.Scene()
        #scene.add(mesh_rend)
        mesh_obj = Visualizer3D.mesh(mesh.trimesh)
        yfov = 2 * math.atan(self._camera_intr.height/(2*self._camera_intr.fy))
        aspect = self._camera_intr.width / self._camera_intr.height
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect, znear=0.001)

        #length=0.2
        #tube_radius=0.005
        #Visualizer3D.arrow(np.array([0.0,0.0,0.0]), length * np.array([1.0,0.0,0.0]), #tube_radius=tube_radius, color=(1, 0, 0))
        #Visualizer3D.arrow(np.array([0.0,0.0,0.0]), length * np.array([0.0,1.0,0.0]), #tube_radius=tube_radius, color=(0, 1, 0))
        #Visualizer3D.arrow(np.array([0.0,0.0,0.0]), length * np.array([0.0,0.0,1.0]), #tube_radius=tube_radius, color=(0, 0, 1))
        #Visualizer3D.pose(T_obj_camera)
        #Visualizer3D.show()

        nc = pyrender.Node(camera=camera, matrix=object_to_camera_poses[0].matrix)
        Visualizer3D._scene.add_node(nc)
        light = pyrender.SpotLight(color=np.ones(3), intensity=1.0,innerConeAngle=np.pi/16.0,outerConeAngle=np.pi/6.0)
        nl = pyrender.Node(light=light, matrix=object_to_camera_poses[0].matrix)
        Visualizer3D._scene.add_node(nl)

        r = pyrender.OffscreenRenderer(viewport_width=self._camera_intr.width, viewport_height=self._camera_intr.height)

        for T_obj_camera in object_to_camera_poses:
            # form projection matrix
            R = T_obj_camera.rotation
            t = T_obj_camera.translation
            P = self._camera_intr.proj_matrix.dot(np.c_[R, t])
    
            # form light props
            #light_props.set_pose(T_obj_camera)
            #light_props_arr = light_props.arr

            # render images for each
            #print(T_obj_camera.matrix)

            #sm = trimesh.creation.uv_sphere(radius=0.01)
            #sm.visual.vertex_colors = [1.0, 1.0, 1.0]
            #tfs = np.tile(np.eye(4), (1, 1, 1))
            #tfs[0,:3,3] = t
            #tfs[1,:3,3] = -1 * T_obj_camera.translation
            #print(tfs[0])
            #point = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            #scene.add(point)

            length=0.2
            tube_radius=0.005
            #T_inv = T_obj_camera.inverse
            #print(T_obj_camera)
            inv_rotation = R.T
            inv_translation = np.dot(-R.T, t)
            rotation_c = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).astype(np.float32)
            
            rot_angle = 0.0 * np.pi / 180.0
            rotation_i = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0.0], [np.sin(rot_angle), np.cos(rot_angle), 0.0], [0.0, 0.0, 1.0]]).astype(np.float32)
            camera_pose = np.dot(np.dot(inv_rotation, rotation_c), rotation_i)
            #Visualizer3D.arrow(t, length * R[:, 0], tube_radius=tube_radius, color=(1, 0, 0))
            #Visualizer3D.arrow(t, length * R[:, 1], tube_radius=tube_radius, color=(0, 1, 0))
            #Visualizer3D.arrow(t, length * R[:, 2], tube_radius=tube_radius, color=(0, 0, 1))
            #Visualizer3D.arrow(inv_translation, length * camera_pose[:, 0], tube_radius=tube_radius, color=(1, 0, 0))
            #Visualizer3D.arrow(inv_translation, length * camera_pose[:, 1], tube_radius=tube_radius, color=(0, 1, 0))
            #Visualizer3D.arrow(inv_translation, length * camera_pose[:, 2], tube_radius=tube_radius, color=(0, 0, 1))
            #Visualizer3D.pose(T_obj_camera)
            #Visualizer3D.show()

            #print(T_obj_camera)
            T_obj_camera.translation = inv_translation
            T_obj_camera.rotation = camera_pose
            #print(T_obj_camera)

            #pyrender.Viewer(scene, use_raymond_lighting=True)

            #scene.add(camera, pose=T_obj_camera.matrix)
            #Visualizer3D.mesh(light, T_mesh_world=T_obj_camera)
            #scene.add(light, pose=T_obj_camera.matrix)

            
            #pts = tm.vertices.copy()
            Visualizer3D._scene.set_pose(nl, pose=T_obj_camera.matrix)
            Visualizer3D._scene.set_pose(nc, pose=T_obj_camera.matrix)
            c, d = r.render(Visualizer3D._scene)
            #c, d = meshrender.render_mesh([P],
            #                              self._camera_intr.height,
            #                              self._camera_intr.width,
            #                              vertex_arr,
            #                              tri_arr,
            #                              norms_arr,
            #                              mat_props_arr,
            #                              light_props_arr,
            #                              enable_lighting,
            #                              debug)
            color_ims.append(c)
            depth_ims.append(d)

            """
            plt.figure()
            plt.subplot(1,2,1)
            plt.axis('off')
            plt.imshow(c)
            plt.subplot(1,2,2)
            plt.axis('off')
            plt.imshow(d, cmap=plt.cm.gray_r)
            plt.show()
            """
            i = i + 1

        r.delete()

        render_stop = time.time()
        logging.debug('Rendering took %.3f sec' %(render_stop - render_start))
        Visualizer3D._scene.clear()
        del r,c,d
        del camera
        del light
        del nc, nl
        del mesh_obj

        #snapshot = tracemalloc.take_snapshot()
        #top_stats = snapshot.statistics('lineno')

        #print("[ Top 10 ]")
        #for stat in top_stats[:10]:
        #    print(stat)

        return color_ims, depth_ims



########


    def images(self, mesh, object_to_camera_poses,
               mat_props=None, light_props=None, enable_lighting=True, debug=False):
        """Render images of the given mesh at the list of object to camera poses.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The mesh to be rendered.
        object_to_camera_poses : :obj:`list` of :obj:`RigidTransform`
            A list of object to camera transforms to render from.
        mat_props : :obj:`MaterialProperties`
            Material properties for the mesh
        light_props : :obj:`MaterialProperties`
            Lighting properties for the scene
        enable_lighting : bool
            Whether or not to enable lighting
        debug : bool
            Whether or not to debug the C++ meshrendering code.

        Returns
        -------
        :obj:`tuple` of `numpy.ndarray`
            A 2-tuple of ndarrays. The first, which represents the color image,
            contains ints (0 to 255) and is of shape (height, width, 3). 
            Each pixel is a 3-ndarray (red, green, blue) associated with a given
            y and x value. The second, which represents the depth image,
            contains floats and is of shape (height, width). Each pixel is a
            single float that represents the depth of the image.
        """
        #tracemalloc.start(15)

        # render for each object to camera pose
        # TODO: clean up interface, use modelview matrix!!!!
        color_ims = []
        depth_ims = []
        render_start = time.time()

        mesh_obj = Visualizer3D.mesh(mesh.trimesh)
        yfov = 2 * math.atan(self._camera_intr.height/(2*self._camera_intr.fy))
        aspect = self._camera_intr.width / self._camera_intr.height
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect, znear=0.001)

        nc = pyrender.Node(camera=camera, matrix=object_to_camera_poses[0].matrix)
        Visualizer3D._scene.add_node(nc)
        light = pyrender.SpotLight(color=np.ones(3), intensity=1.0,innerConeAngle=np.pi/16.0,outerConeAngle=np.pi/6.0)
        nl = pyrender.Node(light=light, matrix=object_to_camera_poses[0].matrix)
        Visualizer3D._scene.add_node(nl)

        r = pyrender.OffscreenRenderer(viewport_width=self._camera_intr.width, viewport_height=self._camera_intr.height)

        for T_obj_camera in object_to_camera_poses:
            # form projection matrix
            R = T_obj_camera.rotation
            t = T_obj_camera.translation


            inv_rotation = R.T
            inv_translation = np.dot(-R.T, t)
            rotation_c = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).astype(np.float32)
            
            rot_angle = 0.0 * np.pi / 180.0
            rotation_i = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0.0], [np.sin(rot_angle), np.cos(rot_angle), 0.0], [0.0, 0.0, 1.0]]).astype(np.float32)
            camera_pose = np.dot(np.dot(inv_rotation, rotation_c), rotation_i)
            T_obj_camera.translation = inv_translation
            T_obj_camera.rotation = camera_pose
            Visualizer3D._scene.set_pose(nl, pose=T_obj_camera.matrix)
            Visualizer3D._scene.set_pose(nc, pose=T_obj_camera.matrix)
            c, d = r.render(Visualizer3D._scene)
            color_ims.append(c)
            depth_ims.append(d)
            gc.collect()

        r.delete()

        render_stop = time.time()
        logging.debug('Rendering took %.3f sec' %(render_stop - render_start))
        Visualizer3D._scene.clear()
        del r,c,d
        del camera
        del light
        del nc, nl
        del mesh_obj

        #snapshot = tracemalloc.take_snapshot()
        #top_stats = snapshot.statistics('traceback')

        #print("[ Top 10 ]")
        #for stat in top_stats[:10]:
        #    print(stat)
        #    for line in stat.traceback.format():
        #        print(line)
        #    print("=====")

        return color_ims, depth_ims

####################

    def images(self, mesh, object_to_camera_poses,
               mat_props=None, light_props=None, enable_lighting=True, debug=False):
        
        def render_mesh(P,im_height,im_width,verts,tris,norms,mat_props,light_props):

            proj_matrices = P

            def draw_object():
                glBegin(GL_TRIANGLES)
                for i in range(tris.shape[0]):
                    for j in range(3):
                        idx = tris[i, j] - 1
                        glNormal3f(norms[idx, 0], norms[idx, 1], norms[idx, 2])
                        glVertex3f(verts[idx, 0], verts[idx, 1], verts[idx, 2])
                glEnd()

            def setup_camera(proj_matrix):
                glMatrixMode(GL_PROJECTION)
                glLoadMatrixf(proj_matrix.T)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()

            def setup_material(mat_props):
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_props[:4])
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_props[4:8])
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_props[8:12])
                glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, mat_props[12])

            def setup_lighting(light_props):
                glEnable(GL_LIGHTING)
                glEnable(GL_LIGHT0)
                glLightfv(GL_LIGHT0, GL_POSITION, light_props[:4])
                glLightfv(GL_LIGHT0, GL_AMBIENT, light_props[4:8])
                glLightfv(GL_LIGHT0, GL_DIFFUSE, light_props[8:12])
                glLightfv(GL_LIGHT0, GL_SPECULAR, light_props[12:])

            def render_scene():
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                setup_camera(proj_matrices)
                setup_lighting(light_props)
                setup_material(mat_props)
                draw_object()
                glutSwapBuffers()

            glutInit()
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
            glutInitWindowSize(im_width, im_height)
            glutCreateWindow("Renderer")
            glEnable(GL_DEPTH_TEST)

            # Set up viewport
            glViewport(0, 0, im_width, im_height)

            # Render scene
            render_scene()

            # Read RGB and Depth image
            c = glReadPixels(0, 0, im_width, im_height, GL_RGB, GL_UNSIGNED_BYTE)
            c = np.frombuffer(c, dtype=np.uint8).reshape(im_height, im_width, 3)[::-1, :]

            d = glReadPixels(0, 0, im_width, im_height, GL_DEPTH_COMPONENT, GL_FLOAT)
            d = np.frombuffer(d, dtype=np.float32).reshape(im_height, im_width)[::-1, :]

            # Cleanup
            glutDestroyWindow(glutGetWindow())

            return c, d
        
        """Render images of the given mesh at the list of object to camera poses.

        Parameters
        ----------
        mesh : :obj:`Mesh3D`
            The mesh to be rendered.
        object_to_camera_poses : :obj:`list` of :obj:`RigidTransform`
            A list of object to camera transforms to render from.
        mat_props : :obj:`MaterialProperties`
            Material properties for the mesh
        light_props : :obj:`MaterialProperties`
            Lighting properties for the scene
        enable_lighting : bool 
            Whether or not to enable lighting
        debug : bool
            Whether or not to debug the C++ meshrendering code.

        Returns
        -------
        :obj:`tuple` of `numpy.ndarray`
            A 2-tuple of ndarrays. The first, which represents the color image,
            contains ints (0 to 255) and is of shape (height, width, 3). 
            Each pixel is a 3-ndarray (red, green, blue) associated with a given
            y and x value. The second, which represents the depth image,
            contains floats and is of shape (height, width). Each pixel is a
            single float that represents the depth of the image.
        """
        # get mesh spec as numpy arrays
        vertex_arr = mesh.vertices
        tri_arr = mesh.triangles.astype(np.int32)
        if mesh.normals is None:
            mesh.compute_vertex_normals()
        norms_arr = mesh.normals

        # set default material properties
        if mat_props is None:
            mat_props = MaterialProperties()
        mat_props_arr = mat_props.arr

        # set default light properties
        if light_props is None:
            light_props = LightingProperties()

        # render for each object to camera pose
        # TODO: clean up interface, use modelview matrix!!!!
        color_ims = []
        depth_ims = []
        render_start = time.time()

        for T_obj_camera in object_to_camera_poses:
            # form projection matrix
            R = T_obj_camera.rotation
            t = T_obj_camera.translation
            P = self._camera_intr.proj_matrix.dot(np.c_[R, t])
    
            # form light props
            light_props.set_pose(T_obj_camera)
            light_props_arr = light_props.arr

            # render images for each
            c, d = render_mesh([P],
                            self._camera_intr.height,
                            self._camera_intr.width,
                            vertex_arr,
                            tri_arr,
                            norms_arr,
                            mat_props_arr,
                            light_props_arr,
                            enable_lighting,
                            debug)
            color_ims.extend(c)
            depth_ims.extend(d)

        render_stop = time.time()
        logging.debug('Rendering took %.3f sec' %(render_stop - render_start))


        return color_ims, depth_ims